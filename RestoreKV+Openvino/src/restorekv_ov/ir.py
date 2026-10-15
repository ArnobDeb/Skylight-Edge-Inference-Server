"""Graph surgery on optimum-exported stateful LLM IRs.

The surgery is applied at load time (read_model is mmap-backed), so no modified
copy of the multi-GB IR is ever written to disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openvino as ov
from openvino import opset13 as ops


@dataclass
class EmbedsModel:
    model: ov.Model
    embed_table: np.ndarray  # [vocab, hidden], dtype of the IR constant (fp16 for *-fp16-ov)

    def embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Host-side embedding lookup, returns float32 [B, T, hidden] like the original Gather."""
        return self.embed_table[np.asarray(token_ids, dtype=np.int64)].astype(np.float32)


def _param(model: ov.Model, name: str) -> ov.Node:
    for p in model.get_parameters():
        if p.get_friendly_name() == name or name in p.get_output_tensor(0).get_names():
            return p
    raise KeyError(f"no parameter {name!r}")


def _find_embedding_gather(input_ids: ov.Node) -> ov.Node:
    """input_ids -> Convert(i32) -> Gather(table, ids, axis=0)."""
    for ti in input_ids.output(0).get_target_inputs():
        node = ti.get_node()
        if node.get_type_name() == "Convert":
            for ti2 in node.output(0).get_target_inputs():
                g = ti2.get_node()
                if g.get_type_name() == "Gather" and ti2.get_index() == 1:
                    return g
    raise RuntimeError("embedding Gather not found next to input_ids")


def _constant_source(node_output: ov.Output) -> ov.Node:
    """Walk back through Convert nodes to the Constant holding the table."""
    node = node_output.get_node()
    while node.get_type_name() == "Convert":
        node = node.input_value(0).get_node()
    if node.get_type_name() != "Constant":
        raise RuntimeError(f"embedding table is not a Constant (got {node.get_type_name()})")
    return node


def make_inputs_embeds_model(model: ov.Model) -> EmbedsModel:
    """Replace `input_ids` with an `inputs_embeds` [B, T, hidden] f32 input.

    All other inputs (attention_mask, position_ids, beam_idx) and the states are unchanged.
    """
    input_ids = _param(model, "input_ids")
    gather = _find_embedding_gather(input_ids)
    table_const = _constant_source(gather.input_value(0))
    embed_table = table_const.get_data()  # zero-copy view into the mmapped .bin

    hidden = gather.get_output_partial_shape(0)[2]
    embeds = ops.parameter(ov.PartialShape([-1, -1, hidden]), gather.get_output_element_type(0), name="inputs_embeds")
    gather.output(0).replace(embeds.output(0))
    # replace() moves the Gather's tensor names onto the new output; keep only ours.
    embeds.get_output_tensor(0).set_names({"inputs_embeds"})

    # Remaining consumers of input_ids only read its shape (batch dim for state init); dims 0/1 are
    # identical for inputs_embeds, and the downstream Gather picks index 0 (batch).
    dead_convert = gather.input_value(1).get_node().get_friendly_name()
    for ti in list(input_ids.output(0).get_target_inputs()):
        consumer = ti.get_node()
        if consumer.get_type_name() == "ShapeOf":
            ti.replace_source_output(embeds.output(0))
        elif consumer.get_friendly_name() == dead_convert:
            continue  # Convert feeding the now-dead Gather
        else:
            raise RuntimeError(f"unexpected input_ids consumer {consumer.get_type_name()} {consumer.get_friendly_name()}")

    model.add_parameters([embeds])
    model.remove_parameter(input_ids)
    model.validate_nodes_and_infer_types()
    return EmbedsModel(model=model, embed_table=embed_table)


LORA_SCALE_INPUT = "lora_scale"


def keep_fp32(node) -> None:
    """Exclude a node from the GPU plugin's f32 -> f16 conversion (legacy `precise` rt_info attribute, converted by
    ConvertLegacyPrecisionAttribute into disable_conversion(node, f16)). The adapters are trained in bf16, whose
    range matches fp32; some LoRA intermediates exceed fp16's 65504 and turn into inf/NaN on the iGPU."""
    node.get_rt_info()["precise_0"] = ""


def add_gated_lora(model: ov.Model, lora: dict[str, tuple[np.ndarray, np.ndarray]], scaling: float) -> int:
    """Add `y += ((lora_scale * scaling * x) @ A.T) @ B.T` to every projection named in `lora`.

    `lora` keys are "layers.<i>.<proj>" (see checkpoint.RestoreCheckpoint), A: [r, in], B: [out, r].
    A scalar f32 input `lora_scale` gates all branches: 1.0 for the restore pass, 0.0 otherwise.
    The gate is applied to the branch INPUT: with 0.0 the branch computes exact zeros. (Gating the branch output
    instead, `up * 0`, turns an fp16 overflow in `up` into inf * 0 = NaN on the GPU, which corrupted the full
    model for some checkpoints even with LoRA "off": 2026-10-09.)
    Returns the number of patched MatMuls.
    """
    gate = ops.parameter(ov.PartialShape([]), ov.Type.f32, name=LORA_SCALE_INPUT)
    gate.get_output_tensor(0).set_names({LORA_SCALE_INPUT})
    gate_scaled = ops.multiply(gate, ops.constant(np.float32(scaling)))

    by_name = {}
    for node in model.get_ordered_ops():
        if node.get_type_name() != "MatMul":
            continue
        name = node.get_friendly_name()  # __module.model.layers.<i>.<self_attn|mlp>.<proj>/ov_ext::linear/MatMul
        if not name.startswith("__module.model.layers."):
            continue
        parts = name.split("/")[0].split(".")
        by_name[f"layers.{parts[3]}.{parts[5]}"] = node

    missing = sorted(set(lora) - set(by_name))
    if missing:
        raise KeyError(f"LoRA targets not found in IR: {missing[:5]} ...")

    for key, (a, b) in lora.items():
        mm = by_name[key]
        assert mm.get_attributes()["transpose_b"], key
        x = mm.input_value(0)
        consumers = list(mm.output(0).get_target_inputs())
        a_c = ops.convert(ops.constant(np.ascontiguousarray(a)), ov.Type.f32)  # [r, in]
        b_c = ops.convert(ops.constant(np.ascontiguousarray(b)), ov.Type.f32)  # [out, r]
        x_gated = ops.multiply(x, gate_scaled)
        down = ops.matmul(x_gated, a_c, transpose_a=False, transpose_b=True)
        up = ops.matmul(down, b_c, transpose_a=False, transpose_b=True)
        for n, tag in ((x_gated, "lora_gate"), (down, "lora_down"), (up, "lora_up"), (a_c, "lora_A"), (b_c, "lora_B")):
            n.set_friendly_name(f"{mm.get_friendly_name()}/{tag}")
            keep_fp32(n)
        out = ops.add(mm.output(0), up)
        out.set_friendly_name(mm.get_friendly_name() + "/lora_add")
        for ti in consumers:
            ti.replace_source_output(out.output(0))

    model.add_parameters([gate])
    model.validate_nodes_and_infer_types()
    return len(lora)


def slice_logits_to_last_token(model: ov.Model) -> None:
    """Compute lm_head only for the last position: logits become [B, 1, vocab].

    Prefill of a 32K chunk would otherwise materialize [1, chunk, 151936] f32 (1.2 GB per 2K chunk).
    """
    lm_head = next(n for n in model.get_ordered_ops()
                   if n.get_type_name() == "MatMul" and n.get_friendly_name().startswith("__module.lm_head"))
    hidden = lm_head.input_value(0)
    last = ops.gather(hidden, ops.constant(np.array([-1], dtype=np.int64)), ops.constant(np.int64(1)))
    lm_head.input(0).replace_source_output(last.output(0))
    model.validate_nodes_and_infer_types()


Q_ROWS_INPUT = "q_rows"


def add_query_outputs(model: ov.Model) -> int:
    """Expose the post-RoPE queries of the last `q_rows` positions of each layer as outputs `q.<layer>`
    ([B, n_heads, q_rows, head_dim] f32). `q_rows` is a runtime i64 scalar input (1 = cheapest).
    Eviction scorers need them: OpenVINO/SnapKV uses the last prompt rows, KVzip the reconstruction rows.
    """
    q_rows = ops.parameter(ov.PartialShape([]), ov.Type.i64, name=Q_ROWS_INPUT)
    q_rows.get_output_tensor(0).set_names({Q_ROWS_INPUT})
    rows_1d = ops.unsqueeze(q_rows, ops.constant(np.int64(0)))
    sdpas = [n for n in model.get_ordered_ops() if n.get_type_name() == "ScaledDotProductAttention"]
    results = []
    for layer, sdpa in enumerate(sdpas):
        q = sdpa.input_value(0)  # [B, H, T, D] after RoPE
        seq_len = ops.gather(ops.shape_of(q, ov.Type.i64), ops.constant(np.array([2], dtype=np.int64)),
                             ops.constant(np.int64(0)))
        start = ops.subtract(seq_len, rows_1d)
        tail = ops.slice(q, start, ops.constant(np.array([np.iinfo(np.int64).max], dtype=np.int64)),
                         ops.constant(np.array([1], dtype=np.int64)), ops.constant(np.array([2], dtype=np.int64)))
        res = ops.result(tail)
        res.get_output_tensor(0).set_names({f"q.{layer}"})
        results.append(res)
    model.add_results(results)
    model.add_parameters([q_rows])
    model.validate_nodes_and_infer_types()
    return len(results)


KV_HEAD_MASK_INPUT = "kv_head_mask"


def add_kv_head_mask(model: ov.Model, n_kv_heads: int) -> int:
    """Add an input `kv_head_mask` (f32, additive: 0 keep, very negative evict) added to every layer's SDPA mask.

    Shape [Bm, n_layers, Hm, Km] with Bm in {1, batch}, Hm in {1, n_kv_heads}, Km <= kv_len (the attention_mask
    length). The mask is zero-padded in-graph from Km up to kv_len, because entries appended after an eviction
    (question / generated tokens) are never evicted; so the host sets it once per eviction, not per decode step.
      * [B, L, n_kv_heads, Km]: per-(row, layer, kv-head, token) eviction by masking ("fake compression", as kvpress
        does), expanded to the GQA query heads (HF repeat_kv order). Each batch row can carry a different
        eviction pattern, so several eviction variants decode in one batch.
      * [1, L, 1, 0]: no eviction. Pads to [1, 1, 1, kv] zeros, which broadcasts without expanding the [B, 1, q, kv]
        causal mask to 32 heads (that expansion costs ~1 GB per layer for a 2K-token prefill chunk).
    """
    hm = ops.parameter(ov.PartialShape([-1, -1, -1, -1]), ov.Type.f32, name=KV_HEAD_MASK_INPUT)
    hm.get_output_tensor(0).set_names({KV_HEAD_MASK_INPUT})
    sdpas = [n for n in model.get_ordered_ops() if n.get_type_name() == "ScaledDotProductAttention"]
    i64 = lambda v: ops.constant(np.array(v, dtype=np.int64))  # noqa: E731
    for layer, sdpa in enumerate(sdpas):
        n_q_heads = sdpa.get_input_partial_shape(0)[1].get_length()
        group = n_q_heads // n_kv_heads
        m = ops.gather(hm, ops.constant(np.int64(layer)), ops.constant(np.int64(1)))           # [Bm, Hm, Km]
        shp = ops.shape_of(m, ov.Type.i64)
        b_m = ops.gather(shp, i64([0]), ops.constant(np.int64(0)))
        h_m = ops.gather(shp, i64([1]), ops.constant(np.int64(0)))
        k_m = ops.gather(shp, i64([2]), ops.constant(np.int64(0)))
        g_m = ops.minimum(h_m, i64([group]))                                                   # G if per-head else 1
        rep = ops.broadcast(ops.unsqueeze(m, i64(2)), ops.concat([b_m, h_m, g_m, k_m], 0))    # [Bm, Hm, Gm, Km]
        per_q = ops.reshape(rep, ops.concat([b_m, i64([-1, 1]), k_m], 0), False)               # [Bm, Hm*Gm, 1, Km]
        orig = sdpa.input_value(3)                                                             # [B, 1, q, kv]
        kv_len = ops.gather(ops.shape_of(orig, ov.Type.i64), i64([3]), ops.constant(np.int64(0)))
        pad_end = ops.concat([i64([0, 0, 0]), ops.subtract(kv_len, k_m)], 0)
        per_q = ops.pad(per_q, i64([0, 0, 0, 0]), pad_end, "constant", ops.constant(np.float32(0)))  # -> kv
        sdpa.input(3).replace_source_output(ops.add(orig, per_q).output(0))
    model.add_parameters([hm])
    model.validate_nodes_and_infer_types()
    return len(sdpas)


def yarn_inv_freq(factor: float, base: float = 1e6, dim: int = 128, original_max: int = 32768,
                  beta_fast: float = 32.0, beta_slow: float = 1.0) -> tuple[np.ndarray, float]:
    """YaRN RoPE (Peng et al. 2023) as in transformers `_compute_yarn_parameters` (Qwen3's recommended long-context
    scaling, rope_type "yarn", original_max_position_embeddings 32768). Returns (inv_freq [dim/2] f64,
    attention_factor), the factor that multiplies cos and sin."""
    import math
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    extrap, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)

    def corr_dim(n_rot: float) -> float:
        return (dim * math.log(original_max / (n_rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    extrap_factor = 1 - ramp
    inv_freq = interp * (1 - extrap_factor) + extrap * extrap_factor
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, attention_factor


def apply_rope_scaling(model: ov.Model, factor: float, base: float = 1e6, original_max: int = 32768) -> float:
    """Patch the optimum-exported RoPE subgraph (rotary_emb: inv_freq Constant [1, dim/2, 1] -> MatMul with
    position_ids -> Cos / Sin) for YaRN: replace inv_freq (stored as f16 in the export; written as exact f32 here)
    and multiply cos and sin by the YaRN attention factor. factor=1 only upgrades inv_freq to f32.
    Returns the attention factor."""
    const = next(n for n in model.get_ordered_ops() if n.get_type_name() == "Constant"
                 and "rotary_emb" in n.get_friendly_name() and tuple(n.get_output_shape(0)) [-1:] == (1,)
                 and len(n.get_output_shape(0)) == 3)
    dim = 2 * int(const.get_output_shape(0)[1])
    inv_freq, att = yarn_inv_freq(factor, base, dim, original_max) if factor > 1 else (
        1.0 / base ** (np.arange(0, dim, 2, dtype=np.float64) / dim), 1.0)
    new = ops.constant(inv_freq.astype(np.float32).reshape(1, dim // 2, 1))
    for ti in list(const.output(0).get_target_inputs()):
        consumer = ti.get_node()
        if consumer.get_type_name() == "Convert":           # f16 -> f32 convert of the original constant
            for ti2 in list(consumer.output(0).get_target_inputs()):
                ti2.replace_source_output(new.output(0))
        else:
            ti.replace_source_output(new.output(0))
    if att != 1.0:
        for n in [n for n in model.get_ordered_ops() if n.get_type_name() in ("Cos", "Sin")
                  and "rotary_emb" in n.get_friendly_name()]:
            consumers = list(n.output(0).get_target_inputs())
            scaled = ops.multiply(n.output(0), ops.constant(np.float32(att)))
            for ti in consumers:
                ti.replace_source_output(scaled.output(0))
    model.validate_nodes_and_infer_types()
    return att
