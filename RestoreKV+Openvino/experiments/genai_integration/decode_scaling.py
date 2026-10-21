"""Experiment 3A/3B: decode scaling inside OpenVINO GenAI, the openvino_benchmarking "experiment 2" design
with KV-cache eviction and RestoreKV.

Fixed input: 4K tokens of a real RULER-4K document (chat template, document, a summarisation request); generated
output 128 ... 16384 tokens with ignore_eos, so every run decodes exactly the requested number of tokens. Three
configurations on the same prompt, GenAI's default (question-aware, continuous) eviction:
    full    no eviction (GenAI baseline)
    ov      OpenVINO H2O eviction (NORM_SUM, SnapKV window 8, start/recent 32, 16-token blocks), --budget of the input
    ov+R    the same + RestoreKV (8 restore tokens before the request, protected; patched GenAI)
Batch 1 for the sweep; --batch8-outputs adds batch-8 points.

Per point (one CSV row) the experiment-2 metrics: TTFT, TPOT, throughput, generate / inference duration (GenAI
PerfMetrics), wall time and wall tokens/s, process RSS before / after / peak, process swap, process disk reads,
system swap-in / swap-out and major page faults (deltas), analytical KV estimate; plus the process's GPU memory
(peak drm-total-gtt) and the KV entries actually held per sequence. The generated token ids are saved for 3B
(agreement with the full-cache output).

    PY=<openvino.genai fork build dir>
    env PYTHONPATH=$PY python experiments/genai_integration/decode_scaling.py \
        --model artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000 --tag qwen3-8b

Output: results/genai_integration/<date>_decode_scaling_<tag>/results.csv, generations.jsonl, config.json (resumable).
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


def vmstat() -> dict[str, int]:
    out = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in ("pswpin", "pswpout", "pgmajfault"):
            out[k] = int(v)
    return out


def proc_status(key: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024          # MB
    return 0.0


def proc_read_bytes() -> int:
    for line in open("/proc/self/io"):
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


class Sampler:
    """Peaks of process GPU memory (GTT), RSS and swap, and the minimum MemAvailable, every 0.2 s."""

    def __init__(self):
        self.peak_gtt = self.peak_rss = self.peak_swap = 0.0
        self.min_avail = 99.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _gtt(self) -> float:
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
            self.peak_gtt = max(self.peak_gtt, self._gtt())
            self.peak_rss = max(self.peak_rss, proc_status("VmRSS"))
            self.peak_swap = max(self.peak_swap, proc_status("VmSwap"))
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
    ap.add_argument("--model", type=Path, required=True, help="GenAI model folder from export_genai_model.py")
    ap.add_argument("--tag", required=True, help="model name for the run folder, e.g. qwen3-8b")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--input-tokens", type=int, default=4096)
    ap.add_argument("--outputs", default="128,256,512,1024,2048,4096,8192,16384")
    ap.add_argument("--batch8-outputs", default="2048,8192")
    ap.add_argument("--configs", default="full,ov,ov+R")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--mem-floor-gb", type=float, default=1.5)
    ap.add_argument("--hybrid", action="store_true",
                    help="question-blind then continuous eviction (patched GenAI, RESTOREKV_QUERY_BLIND=3): the document "
                         "is evicted once before the request is read, then GenAI keeps evicting during the output. "
                         "Configs ov / ov+R only; the full-cache reference is the same prompt as in the default run")
    ap.add_argument("--gpu-limit-gb", type=float, default=26.0, help="skip points whose weights + KV exceed this")
    args = ap.parse_args()
    protect(floor_gb=args.mem_floor_gb)
    if args.hybrid:
        import os
        os.environ["RESTOREKV_QUERY_BLIND"] = "3"                   # read when the pipeline is built

    import openvino as ov
    import openvino_genai as g
    from transformers import AutoTokenizer
    from ruler_eval import load_samples

    cfg = json.loads((args.model / "config.json").read_text())
    n_layers, n_kv = cfg["num_hidden_layers"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_bytes_per_token = 2 * n_layers * n_kv * head_dim * 2           # K and V, f16
    rk = json.loads((args.model / "restorekv.json").read_text())
    restore_ids = np.arange(rk["first_token_id"], rk["first_token_id"] + rk["num_tokens"], dtype=np.int64)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda s: np.array(tok.encode(s, add_special_tokens=False), np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, **({"enable_thinking": False} if "qwen" in args.tag else {}))
    prefix, suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]
    docs = []
    for _, row in load_samples(4096, ["qa_2"], 10):                  # real multi-document QA text
        docs.append(row["context"])
    request = enc("\n\nWrite a long, detailed report about the documents above, covering every document in turn." + suffix)
    ctx = enc(prefix + "\n".join(docs))[: args.input_tokens - len(request)]
    prompts = {"full": np.concatenate([ctx, request]), "ov": np.concatenate([ctx, request]),
               "ov+R": np.concatenate([ctx, restore_ids, request])}
    T = len(prompts["full"])
    n_forced = 0
    if args.hybrid:
        # [document][SEP][request]: the document is prefilled and evicted once, the request is fed as forced tokens;
        # restore tokens 8 tokens before the document end (inside order), as in the question-blind RULER runs
        sep_id = np.array([rk["first_token_id"] + rk["num_tokens"]], np.int64)
        prompts = {"ov": np.concatenate([ctx, sep_id, request]),
                   "ov+R": np.concatenate([ctx[:-8], restore_ids, ctx[-8:], sep_id, request])}
        n_forced = len(request)
    max_cache = max(int(args.budget * T) // args.block * args.block, 32 + 32 + args.block)

    out_dir = ROOT / "results/genai_integration" / f"{args.date}_decode_scaling_{args.tag}{'_hybrid' if args.hybrid else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                      "prompt_tokens": T, "max_cache_size": max_cache,
                                                      "kv_bytes_per_token": kv_bytes_per_token,
                                                      "openvino_genai": g.__file__, "openvino": ov.get_version()},
                                                     indent=2))
    path = out_dir / "results.csv"
    fields = ["model", "config", "budget", "max_cache_size", "input_tokens", "output_tokens", "batch", "status",
              "ttft_ms", "tpot_ms", "throughput_tok_s", "generate_duration_ms", "inference_duration_ms",
              "wall_s", "wall_tok_s", "kv_tokens_per_seq_end", "kv_mb_per_seq_end", "kv_mb_full_cache_equiv",
              "peak_gpu_gtt_gb", "rss_before_mb", "rss_after_mb", "peak_rss_mb", "peak_swap_mb",
              "disk_read_mb", "sys_pswpin", "sys_pswpout", "sys_pgmajfault", "min_mem_available_gb"]
    done = set()
    if path.exists():
        done = {(r["config"], r["output_tokens"], r["batch"]) for r in csv.DictReader(open(path)) if r["status"] == "ok"}
    fout = open(path, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fields)
    if path.stat().st_size == 0:
        w.writeheader()
    gens = open(out_dir / "generations.jsonl", "a")

    outputs = [int(x) for x in args.outputs.split(",") if x]
    points = [(o, 1) for o in outputs] + [(int(o), 8) for o in args.batch8_outputs.split(",") if o]
    props = {"KV_CACHE_PRECISION": "f16", "CACHE_DIR": str(ROOT / "build/ov_cache")}
    if args.device == "GPU":
        props["ACTIVATIONS_SCALE_FACTOR"] = "8"
    for c in args.configs.split(","):
        ids = prompts[c]
        for b in sorted({b for _, b in points}):
            todo = [o for o, bb in points if bb == b and (c, str(o), str(b)) not in done]
            if not todo:
                continue
            weights_gb = (args.model / "openvino_model.bin").stat().st_size / 2**30

            def kv_need(o: int) -> int:     # cache tokens for the whole batch at the end of generation
                return b * (len(ids) + o if c == "full" else max(len(ids), max_cache + args.block))
            base = {"model": args.tag, "config": c, "budget": None if c == "full" else args.budget,
                    "max_cache_size": None if c == "full" else max_cache, "input_tokens": len(ids), "batch": b}
            too_big = [o for o in todo if weights_gb + kv_need(o) * kv_bytes_per_token / 2**30 > args.gpu_limit_gb]
            for o in too_big:                                        # would exhaust the shared memory: record, skip
                need = weights_gb + kv_need(o) * kv_bytes_per_token / 2**30
                w.writerow({**base, "output_tokens": o, "status": f"skipped: weights + KV {need:.1f} GB do not fit"})
                print(f"[{c} b{b} out{o}] skipped ({need:.1f} GB)", flush=True)
            fout.flush()
            todo = [o for o in todo if o not in too_big]
            if not todo:
                continue
            kv_tokens = kv_need(max(todo))
            sc = g.SchedulerConfig()
            sc.cache_size = int(np.ceil(kv_tokens * kv_bytes_per_token / 2**30)) + 1
            sc.max_num_batched_tokens = len(ids) + 64                # one prompt per prefill step, in one chunk
            sc.max_num_seqs = b
            sc.dynamic_split_fuse = False
            if c != "full":
                cec = g.CacheEvictionConfig(32, 32, max_cache, g.AggregationMode.NORM_SUM)
                cec.apply_rotation = False
                cec.snapkv_window_size = 8
                sc.use_cache_eviction = True
                sc.cache_eviction_config = cec
            try:
                pipe = g.ContinuousBatchingPipeline(str(args.model), sc, args.device, props)
            except Exception as e:                                   # noqa: BLE001
                for o in todo:
                    w.writerow({**base, "output_tokens": o, "status": f"pipeline failed: {str(e)[:150]}"})
                fout.flush()
                print(f"[{c} b{b}] pipeline failed: {e}", flush=True)
                continue
            for o in todo:
                gc = g.GenerationConfig()
                gc.max_new_tokens = o + n_forced               # the forced request comes back as generated ids
                gc.ignore_eos = True
                vm0, rd0, rss0 = vmstat(), proc_read_bytes(), proc_status("VmRSS")
                with Sampler() as s:
                    t0 = time.perf_counter()
                    try:
                        res, status = pipe.generate([ov.Tensor(ids[None])] * b, [gc] * b), "ok"
                    except Exception as e:                           # noqa: BLE001
                        res, status = [], f"failed: {str(e)[:150]}"
                    wall = time.perf_counter() - t0
                vm1 = vmstat()
                kv_end = len(ids) + o if c == "full" else min(len(ids) + o, max_cache + args.block)
                row = {**base, "output_tokens": o, "status": status, "wall_s": round(wall, 2),
                       "kv_tokens_per_seq_end": kv_end,
                       "kv_mb_per_seq_end": round(kv_end * kv_bytes_per_token / 2**20, 1),
                       "kv_mb_full_cache_equiv": round((len(ids) + o) * kv_bytes_per_token / 2**20, 1),
                       "peak_gpu_gtt_gb": round(s.peak_gtt, 2), "rss_before_mb": round(rss0),
                       "rss_after_mb": round(proc_status("VmRSS")), "peak_rss_mb": round(s.peak_rss),
                       "peak_swap_mb": round(s.peak_swap), "disk_read_mb": round((proc_read_bytes() - rd0) / 2**20, 1),
                       "sys_pswpin": vm1["pswpin"] - vm0["pswpin"], "sys_pswpout": vm1["pswpout"] - vm0["pswpout"],
                       "sys_pgmajfault": vm1["pgmajfault"] - vm0["pgmajfault"],
                       "min_mem_available_gb": round(s.min_avail, 2)}
                if res:
                    pm = [r.perf_metrics for r in res]
                    n_gen = sum(len(r.m_generation_ids[0]) - n_forced for r in res)
                    row.update(ttft_ms=round(float(np.mean([p.get_ttft().mean for p in pm])), 1),
                               tpot_ms=round(float(np.mean([p.get_tpot().mean for p in pm])), 1),
                               throughput_tok_s=round(float(np.sum([p.get_throughput().mean for p in pm])), 2),
                               generate_duration_ms=round(float(np.mean([p.get_generate_duration().mean for p in pm])), 1),
                               inference_duration_ms=round(float(np.mean([p.get_inference_duration().mean for p in pm])), 1),
                               wall_tok_s=round(n_gen / wall, 2))
                    gens.write(json.dumps({"config": c, "output_tokens": o, "batch": b,
                                           "ids": [int(t) for t in res[0].m_generation_ids[0]][n_forced:]}) + "\n")
                    gens.flush()
                w.writerow(row)
                fout.flush()
                print(f"[{args.tag} {c} b{b} out{o}] {status} ttft {row.get('ttft_ms')} tpot {row.get('tpot_ms')} "
                      f"tok/s {row.get('wall_tok_s')} gtt {row['peak_gpu_gtt_gb']} GB kv {row['kv_mb_per_seq_end']} MB",
                      flush=True)
            del pipe
    fout.close()
    gens.close()


if __name__ == "__main__":
    main()
