"""07: which of our IR edits makes the RestoreKV prototype's prefill slow?

The plain stateful Qwen3-8B IR prefills a 4K prompt in ~8 s on the iGPU (profiling/01), our OVRestoreEngine in ~49 s.
This builds the IR with a chosen subset of the edits from src/restorekv_ov/ir.py, compiles it on the GPU and times
the prefill of one RULER prompt (2048-token chunks). One configuration per process.

    env -u PYTHONPATH python profiling/07_prefill_ablation.py --edits embeds,slice,qout,mask,lora --asf 8 --tokens 4096
Edits: embeds (inputs_embeds input), slice (last-token logits), qout (query outputs), mask (kv_head_mask input),
lora (gated LoRA branches, fp32-pinned), f16lora (gated LoRA branches without fp32 pinning).
Appends a row to results/eviction_study/prefill_ablation.csv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov import ir  # noqa: E402
from restorekv_ov.checkpoint import RestoreCheckpoint  # noqa: E402
from restorekv_ov.safety import protect  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
ap.add_argument("--edits", default="")
ap.add_argument("--asf", type=float, default=0)
ap.add_argument("--tokens", type=int, default=4096, help="prompt length (RULER-4K sample, repeated if needed)")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--tag", default="")
args = ap.parse_args()
protect(floor_gb=1.5)
edits = set(e for e in args.edits.split(",") if e)

core = ov.Core()
model = core.read_model(args.model / "openvino_model.xml")
embed_table = None
if "embeds" in edits:
    em = ir.make_inputs_embeds_model(model)
    model, embed_table = em.model, em.embed_table
if "lora" in edits or "f16lora" in edits:
    ck = RestoreCheckpoint.load(ROOT / "artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz")
    if "f16lora" in edits:
        orig = ir.keep_fp32
        ir.keep_fp32 = lambda node: None
    ir.add_gated_lora(model, ck.lora, ck.scaling)
    if "f16lora" in edits:
        ir.keep_fp32 = orig
if "slice" in edits:
    ir.slice_logits_to_last_token(model)
n_layers = 36
if "qout" in edits:
    n_layers = ir.add_query_outputs(model)
if "mask" in edits:
    ir.add_kv_head_mask(model, 8)
cfg = {"KV_CACHE_PRECISION": "f16"}
if args.asf:
    cfg["ACTIVATIONS_SCALE_FACTOR"] = str(args.asf)
t = time.perf_counter()
compiled = core.compile_model(model, "GPU", cfg)
compile_s = time.perf_counter() - t
req = compiled.create_infer_request()
names = {i.get_any_name() for i in compiled.inputs}

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(args.model)
text = json.loads(open(ROOT / "artifacts/data/ruler/32768/niah_single_1/validation.jsonl").readline())["input"]
ids = np.array(tok.encode(text, add_special_tokens=False)[:args.tokens], np.int64)
T = len(ids)

def prefill_once() -> float:
    req.reset_state()
    t = time.perf_counter()
    pos = 0
    for s in range(0, T, args.chunk):
        part = ids[s:s + args.chunk]
        n = len(part)
        feed = {"attention_mask": np.ones((1, pos + n), np.int64),
                "position_ids": np.arange(pos, pos + n, dtype=np.int64)[None], "beam_idx": np.zeros(1, np.int32)}
        if "inputs_embeds" in names:
            feed["inputs_embeds"] = embed_table[part][None].astype(np.float32)
        else:
            feed["input_ids"] = part[None]
        if ir.Q_ROWS_INPUT in names:
            feed[ir.Q_ROWS_INPUT] = np.int64(8 if s + args.chunk >= T else 1)
        if ir.KV_HEAD_MASK_INPUT in names:
            feed[ir.KV_HEAD_MASK_INPUT] = np.zeros((1, n_layers, 1, 0), np.float32)
        if ir.LORA_SCALE_INPUT in names:
            feed[ir.LORA_SCALE_INPUT] = np.float32(0)
        req.infer(feed)
        pos += n
    return time.perf_counter() - t


warm_s = prefill_once()      # the first inference after compiling includes GPU kernel warm-up
prefill_s = prefill_once()
logits = req.get_tensor("logits").data
row = {"tag": args.tag, "model": args.model.name, "edits": "+".join(sorted(edits)) or "none", "asf": args.asf,
       "tokens": T, "chunk": args.chunk, "compile_s": round(compile_s, 1), "first_prefill_s": round(warm_s, 2),
       "prefill_s": round(prefill_s, 2),
       "last_logit_argmax": int(logits.reshape(-1, logits.shape[-1])[-1].argmax())}
print(row, flush=True)
out = ROOT / "results/eviction_study/prefill_ablation.csv"
new = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
