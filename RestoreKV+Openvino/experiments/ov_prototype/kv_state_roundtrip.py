"""KV state read/write on the stateful IR, and eviction by compaction.

Check A (round-trip): prefill, read all KV states, reset, write them back, decode N tokens.
    Must equal decoding without touching the state (bit-exact).
Check B (compaction == masking): evict a token set D by physically removing it from every layer's K/V,
    keep original positions, decode N tokens (teacher-forced). Reference: full cache with
    attention_mask = 0 on D. Equality shows that keys keep their RoPE positions after compaction and that
    the causal mask follows the KV length, i.e. evict-by-compaction is valid on this IR.

    conda activate arnob
    env -u PYTHONPATH python experiments/ov_prototype/kv_state_roundtrip.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov.engine import OVRestoreEngine  # noqa: E402
from restorekv_ov.runner import load_tokenizer  # noqa: E402


def decode_forced(eng: OVRestoreEngine, first_logits, forced, mask_prefix=None):
    """Teacher-forced decode; returns stacked logits [len(forced)+1, vocab]. mask_prefix: attention_mask for the
    entries already in the cache (used for the masking reference)."""
    outs = [first_logits]
    for tok in forced:
        am = None
        if mask_prefix is not None:
            am = np.concatenate([mask_prefix, np.ones(eng.kv_len - len(mask_prefix) + 1, np.int64)])
        outs.append(eng.forward(eng.embed([tok]), attention_mask=am).logits)
    return np.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--n-decode", type=int, default=8)
    ap.add_argument("--drop-frac", type=float, default=0.5)
    args = ap.parse_args()

    core = ov.Core()
    text = (ROOT / "openvino.genai/README.md").read_text()[:6000]
    prompt = text + "\n\nQuestion: In one sentence, what does OpenVINO GenAI provide?\nAnswer:"
    ids = load_tokenizer(core, args.model)(prompt)[0]
    T = len(ids)
    eng = OVRestoreEngine(args.model, args.device, checkpoint=None, core=core)

    # reference: plain prefill + greedy decode
    eng.reset()
    first = eng.prefill(ids).logits
    ref_tokens = eng.greedy(first, args.n_decode)
    eng.reset()
    ref = decode_forced(eng, eng.prefill(ids).logits, ref_tokens[:-1])

    # A: round-trip
    eng.reset()
    first = eng.prefill(ids).logits
    kv = eng.get_kv()
    eng.set_kv(kv); eng.next_pos = T  # no reset(): GPU ignores set_state after reset_state
    rt = decode_forced(eng, first, ref_tokens[:-1])
    a_diff = float(np.abs(rt - ref).max())

    # B: compaction vs attention_mask=0 on the same set
    rng = np.random.default_rng(0)
    protected = np.zeros(T, bool); protected[:32] = True; protected[-64:] = True   # sink + recent, like OpenVINO
    cand = np.flatnonzero(~protected)
    drop = np.sort(rng.choice(cand, size=int(len(cand) * args.drop_frac), replace=False))
    keep = np.setdiff1d(np.arange(T), drop)

    eng.reset()
    first = eng.prefill(ids).logits      # last-token logits are computed before eviction in both paths
    full = eng.get_kv()
    eng.set_kv([(k[:, keep], v[:, keep]) for k, v in full]); eng.next_pos = T
    compact = decode_forced(eng, first, ref_tokens[:-1])

    mask = np.ones(T, np.int64); mask[drop] = 0
    eng.reset()
    first_m = eng.prefill(ids).logits
    masked = decode_forced(eng, first_m, ref_tokens[:-1], mask_prefix=mask)
    b_diff = float(np.abs(compact[1:] - masked[1:]).max())
    b_agree = float((compact.argmax(-1) == masked.argmax(-1)).mean())
    evict_vs_full = float((compact.argmax(-1) == ref.argmax(-1)).mean())

    metrics = {"prompt_tokens": T, "n_decode": args.n_decode,
               "A_roundtrip_max_abs_logit_diff": a_diff,
               "B_dropped_tokens": int(len(drop)), "B_kept_tokens": int(len(keep)),
               "B_compaction_vs_mask_max_abs_logit_diff": b_diff, "B_compaction_vs_mask_argmax_agreement": b_agree,
               "info_evicted_vs_fullcache_argmax_agreement": evict_vs_full,
               "greedy_tokens": ref_tokens}
    out = ROOT / "results/ov_prototype" / f"{date.today()}_kv_state_roundtrip_{args.model.name}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": str(args.model), "device": args.device, "kv": "f16",
                                                  "drop_frac": args.drop_frac, "openvino": ov.get_version()}, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    ok = a_diff == 0.0 and b_agree == 1.0 and b_diff < 0.1
    print("PASS" if ok else "CHECK", "->", out)


if __name__ == "__main__":
    main()
