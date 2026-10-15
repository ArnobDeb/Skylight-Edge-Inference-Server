"""02: long-context TTFT / TPOT / throughput through openvino_genai's continuous-batching path (PagedAttention),
with and without OpenVINO's built-in KV-cache eviction.

    conda activate arnob
    env -u PYTHONPATH python profiling/02_cb_long_context.py --lengths 4096,16384,32768 --evict 0,1
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> int:
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024


def make_pipe(model: Path, device: str, evict: bool, budget: int, chunk: int, cache_gb: int):
    sc = ov_genai.SchedulerConfig()
    sc.cache_size = cache_gb
    sc.max_num_batched_tokens = chunk  # prefill chunk size
    sc.dynamic_split_fuse = True
    sc.enable_prefix_caching = False
    sc.use_cache_eviction = evict
    if evict:
        sc.cache_eviction_config = ov_genai.CacheEvictionConfig(
            start_size=32, recent_size=128, max_cache_size=budget,
            aggregation_mode=ov_genai.AggregationMode.NORM_SUM, apply_rotation=False)
    return ov_genai.LLMPipeline(str(model), device, scheduler_config=sc, KV_CACHE_PRECISION=ov.Type.f16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--lengths", default="4096,16384,32768")
    ap.add_argument("--evict", default="0,1", help="comma list of 0/1")
    ap.add_argument("--budget", type=int, default=2048, help="max_cache_size (tokens) when evicting")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--cache-gb", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for evict in [bool(int(x)) for x in args.evict.split(",")]:
        pipe = make_pipe(args.model, args.device, evict, args.budget, args.chunk, args.cache_gb)
        gc = ov_genai.GenerationConfig(max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False)
        for n in [int(x) for x in args.lengths.split(",")]:
            ids = rng.integers(1000, 100000, size=(1, n), dtype=np.int64)
            inp = ov_genai.TokenizedInputs(ov.Tensor(ids), ov.Tensor(np.ones_like(ids)))
            t = time.perf_counter()
            try:
                res = pipe.generate(inp, gc)
            except Exception as e:
                rows.append({"evict": evict, "context": n, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                print(rows[-1], flush=True)
                continue
            wall = time.perf_counter() - t
            pm = res.perf_metrics
            rows.append({
                "evict": evict, "budget": args.budget if evict else None, "context": n, "wall_s": wall,
                "ttft_ms": pm.get_ttft().mean, "tpot_ms": pm.get_tpot().mean,
                "decode_throughput_tok_s": pm.get_throughput().mean,
                "prefill_tok_s": n / (pm.get_ttft().mean / 1000), "host_rss_mb": rss_mb(),
            })
            print(json.dumps(rows[-1]), flush=True)
        del pipe

    out = ROOT / "results/profiling" / f"{date.today()}_cb_long_context_{args.model.name}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "model": str(args.model), "openvino": ov.get_version(),
                                                  "openvino_genai": ov_genai.__version__}, indent=2))
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
