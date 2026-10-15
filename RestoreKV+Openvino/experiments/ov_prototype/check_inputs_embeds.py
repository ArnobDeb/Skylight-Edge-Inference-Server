"""The inputs_embeds IR variant must reproduce the input_ids IR.

Runs prefill + N greedy decode steps with the original IR, then replays the same
tokens through the inputs_embeds variant (host-side embedding lookup) and compares
logits. The two models are compiled one after another to stay within RAM.

    env -u PYTHONPATH env/venv-ov/bin/python experiments/ov_prototype/check_inputs_embeds.py \
        [--model ../models/qwen3-8b-fp16-ov] [--device GPU]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov.ir import make_inputs_embeds_model  # noqa: E402
from restorekv_ov.runner import StatefulLM, load_tokenizer  # noqa: E402

PROMPT = (
    "The Skylight edge server keeps a compressed key-value cache for every long document it serves. "
    "When the cache budget is exceeded, the least important blocks are evicted. "
    "Question: what happens when the cache budget is exceeded? Answer:"
)


def run(lm: StatefulLM, ids: np.ndarray, decode_tokens: list[int] | None, n_decode: int, embed=None):
    """Prefill then decode. If decode_tokens is given, teacher-force them; else greedy."""
    lm.reset()
    feed = (lambda x: {"inputs_embeds": embed(x)}) if embed else (lambda x: {"input_ids": x})
    t0 = time.perf_counter()
    logits = [lm.forward(**feed(ids))[:, -1]]
    ttft = time.perf_counter() - t0
    tokens = []
    for i in range(n_decode):
        tok = int(logits[-1].argmax()) if decode_tokens is None else decode_tokens[i]
        tokens.append(tok)
        logits.append(lm.forward(**feed(np.array([[tok]], dtype=np.int64)))[:, -1])
    return np.concatenate(logits), tokens, ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--n-decode", type=int, default=8)
    ap.add_argument("--kv-precision", default="f16")
    args = ap.parse_args()

    core = ov.Core()
    cfg = {"KV_CACHE_PRECISION": args.kv_precision}  # defaults are quantized (CPU u8, GPU dynamic)
    ids = load_tokenizer(core, args.model)(PROMPT)

    m = core.read_model(args.model / "openvino_model.xml")
    lm = StatefulLM(core.compile_model(m, args.device, cfg))
    ref_logits, ref_tokens, ref_ttft = run(lm, ids, None, args.n_decode)
    del lm, m
    gc.collect()

    em = make_inputs_embeds_model(core.read_model(args.model / "openvino_model.xml"))
    lm = StatefulLM(core.compile_model(em.model, args.device, cfg))
    assert "inputs_embeds" in lm.input_names and "input_ids" not in lm.input_names
    new_logits, _, new_ttft = run(lm, ids, ref_tokens, args.n_decode, embed=em.embed)

    diff = np.abs(ref_logits - new_logits)
    metrics = {
        "prompt_tokens": int(ids.shape[1]),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "argmax_agreement": float((ref_logits.argmax(-1) == new_logits.argmax(-1)).mean()),
        "ref_ttft_s": ref_ttft,
        "embeds_ttft_s": new_ttft,
        "greedy_tokens": ref_tokens,
    }
    config = {"model": str(args.model), "device": args.device, "compile_config": cfg, "n_decode": args.n_decode,
              "openvino": ov.get_version()}
    out = ROOT / "results/ov_prototype" / f"{date.today()}_inputs_embeds_equivalence_{args.model.name}_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("PASS" if metrics["argmax_agreement"] == 1.0 and metrics["max_abs_logit_diff"] < 1e-2 else "CHECK", "->", out)


if __name__ == "__main__":
    main()
