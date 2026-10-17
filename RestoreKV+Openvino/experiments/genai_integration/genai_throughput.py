"""decode throughput, latency and memory inside OpenVINO GenAI (ContinuousBatchingPipeline), the
openvino_benchmarking "experiment 2" setup with KV-cache eviction and RestoreKV added.

Fixed input (a RULER document + question, cut to --input-tokens), output 128 .. 2048 tokens with ignore_eos, batch
1 / 4 / 8 identical requests; three configurations per point:
    full    no eviction
    ov      OpenVINO H2O eviction (NORM_SUM, SnapKV window 8, start/recent 32, 16-token blocks), budget --budget
    ov+R    the same + RestoreKV (8 restore tokens in the prompt, protected; patched GenAI)
Eviction keeps each sequence at max_cache_size = budget x input tokens during the whole generation (OpenVINO evicts
again whenever the cache outgrows it).

Per point: TTFT, TPOT, throughput from GenAI's PerfMetrics (mean over the batch's requests), wall-clock tokens/s
(all generated tokens / wall time), process GPU memory (peak drm-total-gtt of this process, sampled every 0.2 s),
peak RSS and the minimum MemAvailable. One pipeline per (configuration, batch); requests run one point at a time.

    PY=<openvino.genai fork build dir>
    env PYTHONPATH=$PY python experiments/genai_integration/genai_throughput.py --input-tokens 4096

Output: results/genai_integration/<date>_genai_throughput_<N>/results.csv + config.json (resumable).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import resource
import sys
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/ov_prototype"))

from restorekv_ov.safety import mem_available_gb, protect  # noqa: E402

KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2      # Qwen3-8B, K and V, f16: 144 KiB


class MemSampler:
    """Peak GPU (GTT) memory of this process from /proc/self/fdinfo, and minimum MemAvailable."""

    def __init__(self):
        self.peak_gtt = 0.0
        self.min_avail = 99.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _gtt_gb(self) -> float:
        tot = 0
        for f in glob.glob("/proc/self/fdinfo/*"):
            try:
                for line in open(f):
                    if line.startswith("drm-total-gtt"):
                        tot += int(line.split()[1])
            except OSError:
                pass
        return tot / 1048576

    def _run(self):
        while not self._stop.is_set():
            self.peak_gtt = max(self.peak_gtt, self._gtt_gb())
            self.min_avail = min(self.min_avail, mem_available_gb())
            time.sleep(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--input-tokens", type=int, default=4096)
    ap.add_argument("--outputs", default="128,512,1024,2048")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--configs", default="full,ov,ov+R")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--weights-gb", type=float, default=15.5, help="GPU memory of the weights, for the fit check")
    ap.add_argument("--gpu-limit-gb", type=float, default=27.0, help="skip points whose weights + KV would exceed this")
    ap.add_argument("--out", default=None)
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--mem-floor-gb", type=float, default=1.0)
    args = ap.parse_args()
    protect(floor_gb=args.mem_floor_gb)

    import openvino as ov
    import openvino_genai as g
    from transformers import AutoTokenizer
    from ruler_eval import load_samples

    rk = json.loads((args.model / "restorekv.json").read_text())
    restore_ids = np.arange(rk["first_token_id"], rk["first_token_id"] + rk["num_tokens"], dtype=np.int64)
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda s: np.array(tok.encode(s, add_special_tokens=False), np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix, suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]
    # document: RULER-4K contexts concatenated until long enough, then cut to the requested length
    docs, n = [], 0
    for _, row in load_samples(4096, None, 10):
        docs.append(row["context"])
        n += len(row["context"]) // 3
        if n > args.input_tokens * 1.5:
            break
    question = enc("\n\nSummarise the text above in detail." + suffix)
    ctx = enc(prefix + "\n".join(docs))[: args.input_tokens - len(question)]
    prompts = {"full": np.concatenate([ctx, question]), "ov": np.concatenate([ctx, question]),
               "ov+R": np.concatenate([ctx, restore_ids, question])}
    T = len(prompts["full"])
    max_cache = max(int(args.budget * T) // args.block * args.block, 32 + 32 + args.block)

    name = args.out or f"genai_throughput_{args.input_tokens // 1024}k"
    out_dir = ROOT / "results/genai_integration" / f"{args.date}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                      "prompt_tokens": T, "max_cache_size": max_cache,
                                                      "openvino_genai": g.__file__, "openvino": ov.get_version()},
                                                     indent=2))
    path = out_dir / "results.csv"
    fields = ["config", "budget", "max_cache_size", "input_tokens", "output_tokens", "batch", "status",
              "ttft_ms", "tpot_ms", "genai_throughput_tok_s", "wall_s", "wall_tok_s", "kv_mb_per_seq",
              "peak_gpu_gtt_gb", "peak_rss_gb", "min_mem_available_gb"]
    done = set()
    if path.exists():
        done = {(r["config"], r["output_tokens"], r["batch"]) for r in csv.DictReader(open(path))}
    fout = open(path, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fields)
    if not done:
        w.writeheader()

    outputs = [int(x) for x in args.outputs.split(",")]
    props = {"KV_CACHE_PRECISION": "f16", "CACHE_DIR": str(ROOT / "build/ov_cache")}
    if args.device == "GPU":
        props["ACTIVATIONS_SCALE_FACTOR"] = "8"
    for cfg in args.configs.split(","):
        ids = prompts[cfg]
        for b in [int(x) for x in args.batches.split(",")]:
            todo = [o for o in outputs if (cfg, str(o), str(b)) not in done]
            if not todo:
                continue
            seq_peak = len(ids) + max(todo)                          # tokens per sequence at the end (full cache)
            # evicted: the whole prompt is cached until the first decode step evicts it down to max_cache_size
            kv_tokens = b * (seq_peak if cfg == "full" else max(len(ids), max_cache + args.block))
            need_gb = kv_tokens * KV_BYTES_PER_TOKEN / 2**30
            base = {"config": cfg, "budget": None if cfg == "full" else args.budget,
                    "max_cache_size": None if cfg == "full" else max_cache, "input_tokens": len(ids), "batch": b}
            if args.weights_gb + need_gb > args.gpu_limit_gb:
                for o in todo:
                    w.writerow({**base, "output_tokens": o, "status": f"skipped (KV {need_gb:.1f} GB does not fit)"})
                fout.flush()
                print(f"[{cfg} b{b}] skipped: KV {need_gb:.1f} GB", flush=True)
                continue
            sc = g.SchedulerConfig()
            sc.cache_size = int(np.ceil(need_gb)) + 1
            sc.max_num_batched_tokens = len(ids) + 64                # one prompt per prefill step, in one chunk
            sc.max_num_seqs = b
            sc.dynamic_split_fuse = False
            if cfg != "full":
                cec = g.CacheEvictionConfig(32, 32, max_cache, g.AggregationMode.NORM_SUM)
                cec.apply_rotation = False
                cec.snapkv_window_size = 8
                sc.use_cache_eviction = True
                sc.cache_eviction_config = cec
            try:
                pipe = g.ContinuousBatchingPipeline(str(args.model), sc, args.device, props)
            except Exception as e:                                   # noqa: BLE001
                for o in todo:
                    w.writerow({**base, "output_tokens": o, "status": f"pipeline failed: {str(e)[:120]}"})
                fout.flush()
                print(f"[{cfg} b{b}] pipeline failed: {e}", flush=True)
                continue
            for o in todo:
                gc = g.GenerationConfig()
                gc.max_new_tokens = o
                gc.ignore_eos = True
                with MemSampler() as mem:
                    t0 = time.perf_counter()
                    try:
                        res = pipe.generate([ov.Tensor(ids[None])] * b, [gc] * b)
                        status = "ok"
                    except Exception as e:                           # noqa: BLE001
                        res, status = [], f"failed: {str(e)[:120]}"
                    wall = time.perf_counter() - t0
                row = {**base, "output_tokens": o, "status": status, "wall_s": round(wall, 2),
                       "peak_gpu_gtt_gb": round(mem.peak_gtt, 2),
                       "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
                       "min_mem_available_gb": round(mem.min_avail, 2),
                       "kv_mb_per_seq": round((len(ids) + o if cfg == "full" else max_cache) * KV_BYTES_PER_TOKEN / 2**20, 1)}
                if res:
                    gen = sum(len(r.m_generation_ids[0]) for r in res)
                    pm = [r.perf_metrics for r in res]
                    row.update(ttft_ms=round(float(np.mean([p.get_ttft().mean for p in pm])), 1),
                               tpot_ms=round(float(np.mean([p.get_tpot().mean for p in pm])), 1),
                               genai_throughput_tok_s=round(float(np.sum([p.get_throughput().mean for p in pm])), 2),
                               wall_tok_s=round(gen / wall, 2))
                w.writerow(row)
                fout.flush()
                print(f"[{cfg} b{b} out{o}] {status} ttft {row.get('ttft_ms')} tpot {row.get('tpot_ms')} "
                      f"wall {row.get('wall_tok_s')} tok/s gtt {row['peak_gpu_gtt_gb']} GB", flush=True)
            del pipe
    fout.close()


if __name__ == "__main__":
    main()
