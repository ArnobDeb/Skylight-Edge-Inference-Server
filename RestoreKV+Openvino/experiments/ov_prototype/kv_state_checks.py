"""correctness of KV-state surgery on the stateful IR (and of the LoRA gate / restore pass).

T1  LoRA-patched IR with lora_scale=0 reproduces the plain IR.
T2  get_kv -> set_kv round trip without changes leaves decoding unchanged.
T3  Eviction by compaction (drop tokens from the state, keep RoPE positions) == masking the same tokens
    with attention_mask zeros on the full cache.
T4  Restore pass appends n entries per layer; LoRA changes them (lora 1 vs 0).

    conda activate arnob
    env -u PYTHONPATH python experiments/ov_prototype/kv_state_checks.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov.engine import OVRestoreEngine  # noqa: E402


def max_diff(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/hf/restorekv/qwen3-8b_restorekv.npz")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    ctx = rng.integers(1000, 30000, size=args.ctx)
    probe = rng.integers(1000, 30000, size=4)  # tokens fed after the context
    metrics = {}

    # ---- T1: plain IR vs LoRA IR at lora_scale = 0 -------------------------------------------------
    plain = OVRestoreEngine(args.model, args.device)
    ref = [plain.prefill(ctx).logits] + [plain.forward(plain.embed([t])).logits for t in probe]
    del plain
    gc.collect()

    eng = OVRestoreEngine(args.model, args.device, checkpoint=args.ckpt)
    got = [eng.prefill(ctx).logits] + [eng.forward(eng.embed([t])).logits for t in probe]
    metrics["T1_lora_gate_off_max_logit_diff"] = max(max_diff(a, b) for a, b in zip(ref, got))
    metrics["T1_argmax_agree"] = all(int(a.argmax()) == int(b.argmax()) for a, b in zip(ref, got))

    # ---- T2: round trip ---------------------------------------------------------------------------
    eng.reset()
    eng.prefill(ctx)
    kv = eng.get_kv()
    eng.set_kv(kv)
    rt = [eng.forward(eng.embed([t])).logits for t in probe]
    metrics["T2_roundtrip_max_logit_diff"] = max(max_diff(a, b) for a, b in zip(got[1:], rt))
    metrics["kv_shape_layer0"] = list(kv[0][0].shape)

    # ---- T3: compaction == attention_mask masking ---------------------------------------------------
    T = len(ctx)
    keep = np.sort(np.concatenate([np.arange(4), 4 + np.sort(rng.choice(T - 4, size=(T - 4) // 2, replace=False))]))
    # A: compaction
    eng.reset()
    eng.prefill(ctx)
    kv = eng.get_kv()
    eng.set_kv([(k[:, keep], v[:, keep]) for k, v in kv])   # next_pos stays T: original positions kept
    comp = [eng.forward(eng.embed([t])).logits for t in probe]
    # B: masking on the full cache
    eng.reset()
    eng.prefill(ctx)
    mask = np.zeros(T, dtype=np.int64)
    mask[keep] = 1
    masked = []
    for t in probe:
        mask = np.concatenate([mask, [1]])
        masked.append(eng.forward(eng.embed([t]), attention_mask=mask).logits)
    metrics["T3_kept_tokens"] = int(len(keep))
    metrics["T3_compaction_vs_mask_max_logit_diff"] = max(max_diff(a, b) for a, b in zip(comp, masked))
    metrics["T3_argmax_agree"] = all(int(a.argmax()) == int(b.argmax()) for a, b in zip(comp, masked))
    metrics["T3_vs_full_cache_max_logit_diff"] = max(max_diff(a, b) for a, b in zip(comp, got[1:]))

    # ---- T4: restore pass --------------------------------------------------------------------------
    eng.reset()
    eng.prefill(ctx)
    eng.restore_pass()
    kv_on = eng.get_kv()
    eng.reset()
    eng.prefill(ctx)
    eng.forward(eng.ckpt.embeddings[None].astype(np.float32), lora=0.0)
    kv_off = eng.get_kv()
    metrics["T4_kv_len_after_restore"] = int(kv_on[0][0].shape[1])
    metrics["T4_next_pos_after_restore"] = int(eng.next_pos)
    metrics["T4_context_kv_unchanged"] = max_diff(kv_on[5][0][:, :T], kv_off[5][0][:, :T])
    metrics["T4_restore_K_lora_effect_l0"] = max_diff(kv_on[0][0][:, T:], kv_off[0][0][:, T:])
    metrics["T4_restore_V_lora_effect_l35"] = max_diff(kv_on[-1][1][:, T:], kv_off[-1][1][:, T:])

    ok = (metrics["T1_argmax_agree"] and metrics["T2_roundtrip_max_logit_diff"] < 1e-3
          and metrics["T3_argmax_agree"] and metrics["T3_compaction_vs_mask_max_logit_diff"] < 5e-2
          and metrics["T4_kv_len_after_restore"] == T + 8 and metrics["T4_context_kv_unchanged"] == 0.0
          and metrics["T4_restore_K_lora_effect_l0"] > 0)
    metrics["PASS"] = bool(ok)
    out = ROOT / "results/ov_prototype" / f"{date.today()}_kv_state_checks_{args.model.name}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": str(args.model), "ckpt": str(args.ckpt),
                                                 "device": args.device, "ctx": args.ctx, "kv": "f16"}, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("PASS" if ok else "CHECK", "->", out)


if __name__ == "__main__":
    main()
