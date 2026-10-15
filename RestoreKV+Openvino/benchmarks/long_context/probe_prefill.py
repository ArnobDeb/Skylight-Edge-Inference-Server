"""Prefill speed / memory probe for the long-context runs: one RULER prompt of the given length, prefilled on the
iGPU with the RestoreKV engine (INT8 weights, YaRN x2 at 64K) at a given chunk size. One config per process (an
OOM can abort the process). Appends a row to results/eviction_study/long_context_prefill_probe.csv.
    env -u PYTHONPATH python benchmarks/long_context/probe_prefill.py --length 32768 --chunk 2048
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov.engine import OVRestoreEngine  # noqa: E402
from restorekv_ov.safety import mem_available_gb, protect  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, required=True)
ap.add_argument("--chunk", type=int, required=True)
ap.add_argument("--kv", default="f16")
ap.add_argument("--model", type=Path, default=ROOT / "artifacts/models/qwen3-8b-int8-ov")
args = ap.parse_args()
protect(floor_gb=1.5)
out = ROOT / "results/eviction_study/long_context_prefill_probe.csv"
row = {"length": args.length, "chunk": args.chunk, "kv_precision": args.kv, "status": "started"}
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(args.model)
text = json.loads(open(ROOT / f"artifacts/data/ruler/{args.length}/niah_single_1/validation.jsonl").readline())["input"]
ids = np.array(tok.encode(text, add_special_tokens=False), np.int64)
eng = OVRestoreEngine(args.model, "GPU", checkpoint=ROOT / "artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz",
                      kv_precision=args.kv, core=ov.Core(), compile_config={"ACTIVATIONS_SCALE_FACTOR": "8"},
                      rope_factor=2.0 if args.length > 40000 else None)
min_mem = mem_available_gb()
t = time.perf_counter()
for s in range(0, len(ids), args.chunk):
    eng.forward(eng.embed(ids[s:s + args.chunk]), q_rows=8 if s + args.chunk >= len(ids) else 0)
    min_mem = min(min_mem, mem_available_gb())
row.update(status="ok", tokens=len(ids), prefill_s=round(time.perf_counter() - t, 1), min_mem_available_gb=round(min_mem, 1))
new = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(row, flush=True)
