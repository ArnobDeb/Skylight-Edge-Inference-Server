"""01: chunked prefill to long contexts on one device; per-chunk latency, decode TPOT at checkpoints, memory.

Positions past the model's trained range give meaningless outputs, but timing and memory are valid.

    env -u PYTHONPATH python profiling/01_long_context_scaling.py --max-len 65536 --chunk 2048
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> int:
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--kv-precision", default="f16")
    ap.add_argument("--checkpoints", default="4096,8192,16384,32768,65536")
    ap.add_argument("--n-decode", type=int, default=8)
    args = ap.parse_args()
    checkpoints = {int(x) for x in args.checkpoints.split(",")}

    core = ov.Core()
    cfg = {"KV_CACHE_PRECISION": args.kv_precision}
    req = core.compile_model(args.model / "openvino_model.xml", args.device, cfg).create_infer_request()

    def step(t0: int, n: int):
        req.infer({
            "input_ids": np.full((1, n), 1000, np.int64),
            "attention_mask": np.ones((1, t0 + n), np.int64),
            "position_ids": np.arange(t0, t0 + n, dtype=np.int64)[None],
            "beam_idx": np.zeros(1, np.int32),
        })

    chunks, at_ckpt = [], []
    ctx, extra, prefill_total = 0, 0, 0.0  # ctx = prompt tokens fed; extra = decode tokens appended at checkpoints
    try:
        while ctx < args.max_len:
            n = min(args.chunk, args.max_len - ctx)
            t = time.perf_counter(); step(ctx + extra, n); dt = time.perf_counter() - t
            ctx += n; prefill_total += dt
            chunks.append({"context": ctx, "chunk_s": dt, "chunk_tok_s": n / dt})
            if ctx in checkpoints:
                ts = []
                for _ in range(args.n_decode):
                    t = time.perf_counter(); step(ctx + extra, 1); ts.append(time.perf_counter() - t)
                    extra += 1
                rec = {"context": ctx, "prefill_cum_s": prefill_total, "prefill_avg_tok_s": ctx / prefill_total,
                       "last_chunk_tok_s": n / dt, "tpot_ms": 1000 * float(np.median(ts[1:])), "host_rss_mb": rss_mb()}
                at_ckpt.append(rec)
                print(json.dumps(rec), flush=True)
    except Exception as e:  # out-of-memory etc.
        print("STOPPED at", ctx, type(e).__name__, str(e)[:300])
        at_ckpt.append({"stopped_at": ctx, "error": f"{type(e).__name__}: {str(e)[:300]}"})

    out = ROOT / "results/profiling" / f"{date.today()}_long_context_scaling_{args.model.name}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "model": str(args.model), "openvino": ov.get_version(),
                                                  "compile_config": cfg}, indent=2))
    (out / "metrics.json").write_text(json.dumps({"checkpoints": at_ckpt, "chunks": chunks}, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
