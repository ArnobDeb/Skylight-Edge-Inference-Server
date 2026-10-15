"""03: isolate the GPU failure of ContinuousBatching + cache eviction for Qwen3.

Each case runs in a fresh subprocess (a GPU fault can poison the OpenCL context).

    env -u PYTHONPATH python profiling/03_gpu_eviction_diag.py --model ../models/qwen3-4b-fp16-ov
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASE = r'''
import sys, json, numpy as np, openvino as ov, openvino_genai as g
model, n, chunk, evict, snap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5])
sc = g.SchedulerConfig(); sc.cache_size = 6; sc.max_num_batched_tokens = chunk; sc.dynamic_split_fuse = True
sc.enable_prefix_caching = False; sc.use_cache_eviction = evict
if evict:
    sc.cache_eviction_config = g.CacheEvictionConfig(start_size=32, recent_size=128, max_cache_size=1024,
        aggregation_mode=g.AggregationMode.NORM_SUM, apply_rotation=False, snapkv_window_size=snap)
p = g.LLMPipeline(model, "GPU", scheduler_config=sc, KV_CACHE_PRECISION=ov.Type.f16)
ids = np.random.default_rng(0).integers(1000, 30000, size=(1, n), dtype=np.int64)
try:
    r = p.generate(g.TokenizedInputs(ov.Tensor(ids), ov.Tensor(np.ones_like(ids))), g.GenerationConfig(max_new_tokens=4, min_new_tokens=4))
    print("RESULT", json.dumps({"ok": True, "ttft_ms": r.perf_metrics.get_ttft().mean, "tpot_ms": r.perf_metrics.get_tpot().mean}))
except Exception as e:
    print("RESULT", json.dumps({"ok": False, "error": " ".join(str(e).split())[-220:]}))
'''

# (prompt_len, chunk, evict, snapkv_window)
DEFAULT_CASES = [
    (1900, 2048, 1, 8),    # single chunk, PREFILL stage only           (known OK on 4B)
    (2100, 2048, 1, 8),    # 2 chunks, second MIXED chunk is tiny (52 tok)
    (1024, 256, 1, 8),     # many small MIXED chunks, short context
    (4096, 2048, 1, 8),    # known FAIL
    (4096, 2048, 1, 0),    # SnapKV off: scores from every chunk
    (8192, 8192, 1, 8),    # single chunk, PREFILL, 8K
    (16384, 16384, 1, 8),  # single chunk, PREFILL, 16K (8B: FAIL)
    (16384, 16384, 0, 8),  # same, eviction off: is single-chunk 16K itself the problem?
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-4b-fp16-ov")
    args = ap.parse_args()
    rows = []
    for n, chunk, evict, snap in DEFAULT_CASES:
        proc = subprocess.run([sys.executable, "-c", CASE, str(args.model), str(n), str(chunk), str(evict), str(snap)],
                              capture_output=True, text=True, timeout=1800)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"ok": False, "error": f"process died rc={proc.returncode}: "
                                                 + " ".join(proc.stderr.split())[-220:]}
        rows.append({"prompt": n, "chunk": chunk, "evict": bool(evict), "snapkv": snap, **res})
        print(json.dumps(rows[-1]), flush=True)
    out = ROOT / "results/profiling" / f"{date.today()}_gpu_eviction_diag_{args.model.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
