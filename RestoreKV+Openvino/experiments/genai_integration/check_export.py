"""is the exported GenAI model (s4_1) the same computation as the Python prototype?

Three phases, each in its own process (two 8B models do not fit together):
    base    original IR, plain token ids:   [document][question]                -> logits A
    export  exported IR, plain token ids:   [document][question]                -> logits B  (must equal A: LoRA gate 0)
            exported IR, restore ids:       [document][V..V+7][question]        -> logits C, restore-token KV
    proto   OVRestoreEngine (prototype)  : prefill, restore_pass(), question   -> logits D, restore-token KV
    compare B vs A, C vs D and the restore-token keys/values

No eviction here: this checks only that the restore pass inside the prompt reproduces the prototype's restore pass.

    env -u PYTHONPATH python experiments/genai_integration/check_export.py --phase base
    env -u PYTHONPATH python experiments/genai_integration/check_export.py --phase export
    env -u PYTHONPATH python experiments/genai_integration/check_export.py --phase proto
    env -u PYTHONPATH python experiments/genai_integration/check_export.py --phase compare
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/ov_prototype"))

LAYERS = (0, 17, 35)        # layers whose restore-token KV is compared


def prompt(model_dir: Path, n_per_task: int, idx: int):
    from transformers import AutoTokenizer
    from ruler_eval import load_samples
    tok = AutoTokenizer.from_pretrained(model_dir)
    enc = lambda s: np.array(tok.encode(s, add_special_tokens=False), np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix, suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]
    _, row = load_samples(4096, None, n_per_task)[idx]
    return enc(prefix + row["context"]), enc(row["question"] + suffix + row["answer_prefix"])


def compile_request(model_xml: Path, device: str):
    import openvino as ov
    core = ov.Core()
    cfg = {"KV_CACHE_PRECISION": "f16", "ACTIVATIONS_SCALE_FACTOR": "8"} if device == "GPU" else {}
    return core.compile_model(core.read_model(model_xml), device, cfg).create_infer_request()


def run_stateful(req, parts: list[np.ndarray], chunk: int = 2048):
    """Plain OpenVINO runtime on a stateful IR: feed token-id parts in order from an empty cache, return last logits."""
    req.reset_state()
    pos, logits = 0, None
    for part in parts:
        for s in range(0, len(part), chunk):
            ids = part[s:s + chunk][None]
            n = ids.shape[1]
            res = req.infer({"input_ids": ids, "attention_mask": np.ones((1, pos + n), np.int64),
                             "position_ids": np.arange(pos, pos + n, dtype=np.int64)[None],
                             "beam_idx": np.zeros(1, np.int32)})
            logits = req.get_tensor("logits").data[0, -1].astype(np.float32)
            pos += n
    return logits


def state_kv(req, start: int, n: int) -> dict[str, np.ndarray]:
    out = {}
    for st in req.query_state():
        m = re.search(r"past_key_values\.(\d+)\.(key|value)", st.name)
        if m and int(m.group(1)) in LAYERS:
            out[f"{m.group(1)}_{m.group(2)}"] = st.state.data[0][:, start:start + n].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["base", "export", "proto", "compare"], required=True)
    ap.add_argument("--base", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--export", type=Path, default=ROOT / "artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--sample", type=int, default=0, help="index into the round-robin RULER-4K order")
    ap.add_argument("--out", type=Path, default=ROOT / "results/genai_integration/2026-10-17_export_check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase != "compare":
        from restorekv_ov.safety import protect
        protect()
    ctx, q = prompt(args.base, 10, args.sample)
    T = len(ctx)

    if args.phase == "base":
        a = run_stateful(compile_request(args.base / "openvino_model.xml", args.device), [ctx, q])
        np.savez(args.out / "base.npz", logits=a)
    elif args.phase == "export":
        rk = json.loads((args.export / "restorekv.json").read_text())
        r_ids = np.arange(rk["first_token_id"], rk["first_token_id"] + rk["num_tokens"], dtype=np.int64)
        req = compile_request(args.export / "openvino_model.xml", args.device)   # one compiled model, two runs
        b = run_stateful(req, [ctx, q])
        c = run_stateful(req, [ctx, r_ids, q])
        np.savez(args.out / "export.npz", logits_plain=b, logits_restore=c, **state_kv(req, T, len(r_ids)))
    elif args.phase == "proto":
        from restorekv_ov.engine import OVRestoreEngine
        ck = ROOT / "artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz"
        eng = OVRestoreEngine(args.base, args.device, checkpoint=ck, compile_config={"ACTIVATIONS_SCALE_FACTOR": "8"})
        eng.prefill(ctx)
        eng.restore_pass()
        n = eng.n_restore
        kv = eng.get_kv()
        d = eng.forward(eng.embed(q[None])).logits[0].astype(np.float32)
        restore = {f"{i}_{name}": (k if name == "key" else v)[:, T:T + n]
                   for i, (k, v) in enumerate(kv) if i in LAYERS for name in ("key", "value")}
        np.savez(args.out / "proto.npz", logits=d, **restore)
    else:
        base, exp, proto = (np.load(args.out / f"{p}.npz") for p in ("base", "export", "proto"))

        def cmp(x, y):
            return {"max_abs_diff": float(np.abs(x - y).max()), "argmax_equal": bool(x.argmax() == y.argmax()),
                    "top5_overlap": len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5]))}
        rep = {"context_tokens": T,
               "plain_ids_exported_vs_original": cmp(exp["logits_plain"], base["logits"]),
               "restore_in_prompt_vs_prototype_restore_pass": cmp(exp["logits_restore"], proto["logits"]),
               "restore_vs_no_restore (should differ)": cmp(exp["logits_restore"], exp["logits_plain"])}
        for key in proto.files:
            if key != "logits":
                x, y = exp[key], proto[key]
                rep[f"restore_kv_layer{key}"] = {"max_abs_diff": float(np.abs(x - y).max()),
                                                 "rel_diff": float(np.abs(x - y).max() / (np.abs(y).max() + 1e-9))}
        (args.out / "metrics.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
