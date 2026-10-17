"""export a RestoreKV model folder that OpenVINO GenAI (ContinuousBatchingPipeline) can load as is.

GenAI's pipeline only takes token ids and applies LoRA to the whole batch, while RestoreKV needs 8 learned input
vectors and a LoRA adapter that is on for those 8 positions only. Both are built into the graph here, so the C++
pipeline needs no new inputs:

  * restore tokens as new token ids: ids V .. V+7 (V = rows of the embedding table) look up the learned restore
    embeddings, every other id the normal table. The tokenizer and the LM head are unchanged.
  * token-gated LoRA: every LoRA branch is multiplied, per token, by (id >= V). A prompt laid out as
    [document][V..V+7][question] then runs the restore pass inside the normal prefill: the 8 restore tokens see the
    full document with the adapter on, all other tokens run with the adapter off (exact zeros in the branch).
  * rt_info "restorekv/first_token_id" and "restorekv/num_tokens" tell the patched GenAI (build/genai-restorekv)
    which tokens to protect from eviction.

Usage:
    env -u PYTHONPATH python experiments/genai_integration/export_genai_model.py \
        --model ../models/qwen3-8b-fp16-ov --ckpt artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz \
        --out artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from restorekv_ov.checkpoint import RestoreCheckpoint  # noqa: E402
from restorekv_ov.ir import _find_embedding_gather, _param, keep_fp32  # noqa: E402
from restorekv_ov.safety import protect  # noqa: E402


def add_restore_tokens(model: ov.Model, embeddings: np.ndarray) -> tuple[int, ov.Output]:
    """Ids >= V take the learned restore embeddings. Returns V and the per-token gate (f32, ids shape + [1])."""
    gather = _find_embedding_gather(_param(model, "input_ids"))
    ids = gather.input_value(1)                                    # input_ids converted to i32
    vocab = gather.get_input_partial_shape(0)[0].get_length()
    n = embeddings.shape[0]
    consumers = list(gather.output(0).get_target_inputs())

    first = ops.constant(np.int32(vocab))
    is_restore = ops.greater_equal(ids, first)
    gather.input(1).replace_source_output(ops.minimum(ids, ops.constant(np.int32(vocab - 1))).output(0))
    r_idx = ops.minimum(ops.maximum(ops.subtract(ids, first), ops.constant(np.int32(0))), ops.constant(np.int32(n - 1)))
    table = ops.constant(np.ascontiguousarray(embeddings.astype(np.float32)))
    keep_fp32(table)
    r_emb = ops.convert(ops.gather(table, r_idx, ops.constant(np.int32(0))), gather.get_output_element_type(0))
    mask = ops.unsqueeze(is_restore, ops.constant(np.int32(-1)))
    emb = ops.select(mask, r_emb, gather.output(0))
    emb.set_friendly_name("restorekv/embedding_select")
    for ti in consumers:
        ti.replace_source_output(emb.output(0))
    gate = ops.convert(mask, ov.Type.f32)
    gate.set_friendly_name("restorekv/lora_gate")
    return vocab, gate.output(0)


def add_token_gated_lora(model: ov.Model, lora: dict, scaling: float, gate: ov.Output) -> int:
    """Same LoRA branches as ir.add_gated_lora, but gated per token instead of by a scalar input."""
    gate_scaled = ops.multiply(gate, ops.constant(np.float32(scaling)))
    by_name = {}
    for node in model.get_ordered_ops():
        name = node.get_friendly_name()
        if node.get_type_name() == "MatMul" and name.startswith("__module.model.layers."):
            parts = name.split("/")[0].split(".")
            by_name[f"layers.{parts[3]}.{parts[5]}"] = node
    missing = sorted(set(lora) - set(by_name))
    if missing:
        raise KeyError(f"LoRA targets not found in IR: {missing[:5]} ...")
    for key, (a, b) in lora.items():
        mm = by_name[key]
        x = mm.input_value(0)
        consumers = list(mm.output(0).get_target_inputs())
        a_c = ops.convert(ops.constant(np.ascontiguousarray(a)), ov.Type.f32)
        b_c = ops.convert(ops.constant(np.ascontiguousarray(b)), ov.Type.f32)
        x_gated = ops.multiply(x, gate_scaled)       # gate on the branch input: exact zeros when off
        down = ops.matmul(x_gated, a_c, transpose_a=False, transpose_b=True)
        up = ops.matmul(down, b_c, transpose_a=False, transpose_b=True)
        for n, tag in ((x_gated, "lora_gate"), (down, "lora_down"), (up, "lora_up"), (a_c, "lora_A"), (b_c, "lora_B")):
            n.set_friendly_name(f"{mm.get_friendly_name()}/{tag}")
            keep_fp32(n)
        out = ops.add(mm.output(0), up)
        out.set_friendly_name(mm.get_friendly_name() + "/lora_add")
        for ti in consumers:
            ti.replace_source_output(out.output(0))
    return len(lora)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    protect()

    ck = RestoreCheckpoint.load(args.ckpt)
    model = ov.Core().read_model(args.model / "openvino_model.xml")
    vocab, gate = add_restore_tokens(model, ck.embeddings)
    n_lora = add_token_gated_lora(model, ck.lora, ck.scaling, gate)
    model.validate_nodes_and_infer_types()
    n = int(ck.embeddings.shape[0])
    model.set_rt_info(str(vocab), ["restorekv", "first_token_id"])
    model.set_rt_info(str(n), ["restorekv", "num_tokens"])
    model.set_rt_info(args.ckpt.name, ["restorekv", "checkpoint"])

    args.out.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, args.out / "openvino_model.xml", compress_to_fp16=False)
    for f in args.model.iterdir():                    # tokenizer, detokenizer and configs
        if f.is_file() and not f.name.startswith("openvino_model."):
            shutil.copy2(f, args.out / f.name)
    (args.out / "restorekv.json").write_text(json.dumps(
        {"first_token_id": vocab, "num_tokens": n, "checkpoint": str(args.ckpt), "base_model": str(args.model),
         "lora_branches": n_lora, "scaling": ck.scaling}, indent=2))
    print(f"exported to {args.out}: restore ids {vocab}..{vocab + n - 1}, {n_lora} LoRA branches")


if __name__ == "__main__":
    main()
