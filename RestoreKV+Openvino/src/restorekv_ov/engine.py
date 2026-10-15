"""RestoreKV on the plain OpenVINO runtime: one stateful IR, explicit KV / position bookkeeping.

IR = optimum stateful export + inputs_embeds + lora_scale-gated LoRA + last-token logits + per-layer query
tails + per-(row, layer, kv-head, token) eviction mask (see ir.py). One compiled model, one KV state, so there is
no state copy between the restore pass (LoRA on) and decoding (LoRA off).

Position rules (paper / kvpress RestoreKVPress): context at 0..T-1, restore tokens at T..T+n-1 attending to the
full cache, then eviction keeps original RoPE positions (keys are stored post-RoPE), queries continue at T+n.
The causal mask is built from the KV state length, so all rows and layers hold the same number of entries;
eviction is expressed by the additive kv_head_mask (exact for quality, like kvpress's "fake compression").

Batching: decoding is memory-bandwidth bound (every step re-reads the weights), so several eviction variants of
the same sample are decoded together as batch rows (`decode_variants`), each with its own mask and positions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov

from .checkpoint import RestoreCheckpoint
from .ir import (KV_HEAD_MASK_INPUT, LORA_SCALE_INPUT, Q_ROWS_INPUT, add_gated_lora, add_kv_head_mask, add_query_outputs,
                 apply_rope_scaling, make_inputs_embeds_model, slice_logits_to_last_token)

_STATE_RE = re.compile(r"past_key_values\.(\d+)\.(key|value)")
KV = list[tuple[np.ndarray, np.ndarray]]   # per layer (K, V), each [kv_heads, len, head_dim] (K post-RoPE)


@dataclass
class StepOutput:
    logits: np.ndarray                  # [B, vocab] for the last fed position of each row
    queries: list[np.ndarray] | None    # per layer [B, n_heads, q_rows, head_dim] (post-RoPE), if requested


@dataclass
class Variant:
    """One decoding row: which context entries are evicted, and whether the restore entries are visible."""
    name: str
    evict_ctx: np.ndarray | None        # bool [n_layers, n_kv_heads, T]; None = keep all
    restore: bool = False


class OVRestoreEngine:
    MASKED = np.float32(np.finfo(np.float32).min / 4)

    def __init__(self, model_dir: Path, device: str = "GPU", checkpoint: Path | None = None,
                 kv_precision: str = "f16", core: ov.Core | None = None, compile_config: dict | None = None,
                 rope_factor: float | None = None):
        """rope_factor: None = model as exported; >1 = YaRN scaling (e.g. 2.0 for 64K on Qwen3's 32K native
        RoPE); 1.0 = only replace the exported f16 inv_freq by exact f32 values."""
        core = core or ov.Core()
        em = make_inputs_embeds_model(core.read_model(Path(model_dir) / "openvino_model.xml"))
        self.rope_attention_factor = 1.0
        if rope_factor is not None:
            self.rope_attention_factor = apply_rope_scaling(em.model, rope_factor)
        self.embed_table = em.embed_table
        self.ckpt = RestoreCheckpoint.load(checkpoint) if checkpoint else None
        if self.ckpt is not None:
            add_gated_lora(em.model, self.ckpt.lora, self.ckpt.scaling)
        slice_logits_to_last_token(em.model)
        self.n_layers = add_query_outputs(em.model)
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.n_q_heads = cfg["num_attention_heads"]
        self.head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        add_kv_head_mask(em.model, self.n_kv_heads)
        self.compiled = core.compile_model(em.model, device, {"KV_CACHE_PRECISION": kv_precision, **(compile_config or {})})
        self.req = self.compiled.create_infer_request()
        self.input_names = {i.get_any_name() for i in self.compiled.inputs}
        self._clear(batch=1, kv_len=0)

    # ---- bookkeeping ------------------------------------------------------------------------------
    def _clear(self, batch: int, kv_len: int):
        self.batch = batch
        self.kv_len = kv_len
        self.next_pos = np.full(batch, kv_len, np.int64)
        # additive mask over the entries present at the last eviction; later entries are padded with 0 in-graph
        self.head_mask = np.zeros((batch, self.n_layers, self.n_kv_heads, 0), np.float32)
        self.mask_active = False

    def reset(self):
        self.req.reset_state()
        self._clear(1, 0)

    def fresh_request(self):
        """Drop the infer request (its KV states and intermediate GPU buffers) and start a new, empty one. Between long
        documents this frees the previous document's device memory: reusing the request left ~25 GB in use and made
        the next 32K prefill 2.5-4.5x slower (memory pressure on the shared-RAM iGPU)."""
        import gc
        self.req = None
        gc.collect()
        self.req = self.compiled.create_infer_request()
        self._clear(1, 0)

    def mask_entries(self, evict: np.ndarray) -> None:
        """Evict by masking: evict is bool [B, n_layers, n_kv_heads, kv_len] (or without B for batch 1), over the
        current cache entries. Replaces any previous eviction mask."""
        evict = evict.reshape(self.batch, self.n_layers, self.n_kv_heads, self.kv_len)
        self.head_mask = np.where(evict, self.MASKED, np.float32(0)).astype(np.float32)
        self.mask_active = bool(evict.any())

    def kept_entries(self) -> np.ndarray:
        """bool [B, n_layers, n_kv_heads, kv_len]: entries currently visible to attention."""
        pad = self.kv_len - self.head_mask.shape[3]
        return np.pad(self.head_mask, ((0, 0), (0, 0), (0, 0), (0, pad))) == 0

    def embed(self, ids) -> np.ndarray:
        """ids [t] or [B, t] -> [B, t, hidden] f32."""
        ids = np.asarray(ids, dtype=np.int64)
        return self.embed_table[ids.reshape(1, -1) if ids.ndim == 1 else ids].astype(np.float32)

    # ---- stepping ---------------------------------------------------------------------------------
    def forward(self, embeds: np.ndarray, positions: np.ndarray | None = None, lora: float = 0.0,
                q_rows: int = 0, attention_mask: np.ndarray | None = None) -> StepOutput:
        """Feed [B, t, hidden] embeddings (B must equal the current batch). positions default to next_pos+0..t-1."""
        b, t = embeds.shape[0], embeds.shape[1]
        assert b == self.batch, (b, self.batch)
        if positions is None:
            positions = self.next_pos[:, None] + np.arange(t, dtype=np.int64)[None]
        positions = positions.reshape(b, t)
        if attention_mask is None:
            attention_mask = np.ones((b, self.kv_len + t), dtype=np.int64)
        mask_in = self.head_mask if self.mask_active else np.zeros((1, self.n_layers, 1, 0), np.float32)
        feed = {KV_HEAD_MASK_INPUT: mask_in, "inputs_embeds": embeds, "position_ids": positions,
                "attention_mask": attention_mask.reshape(b, -1), "beam_idx": np.arange(b, dtype=np.int32),
                Q_ROWS_INPUT: np.int64(max(q_rows, 1))}
        if LORA_SCALE_INPUT in self.input_names:
            feed[LORA_SCALE_INPUT] = np.float32(lora)
        elif lora:
            raise RuntimeError("engine was built without a RestoreKV checkpoint")
        self.req.infer(feed)
        self.kv_len += t
        self.next_pos = positions.max(axis=1) + 1
        logits = self.req.get_tensor("logits").data[:, -1].copy()
        queries = None
        if q_rows > 0:
            queries = [self.req.get_tensor(f"q.{i}").data.copy() for i in range(self.n_layers)]
        return StepOutput(logits, queries)

    def prefill(self, ids: np.ndarray, chunk: int = 2048, q_rows_last: int = 0) -> StepOutput:
        ids = np.asarray(ids).reshape(-1)
        out = None
        for s in range(0, len(ids), chunk):
            part = ids[s:s + chunk]
            last = s + chunk >= len(ids)
            out = self.forward(self.embed(part), q_rows=min(q_rows_last, len(part)) if last else 0)
        return out

    # ---- KV state access ------------------------------------------------------------------------
    def _states(self) -> dict[tuple[int, str], ov.VariableState]:
        states = {}
        for st in self.req.query_state():
            m = _STATE_RE.search(st.name)
            if m:
                states[(int(m.group(1)), m.group(2))] = st
        assert len(states) == 2 * self.n_layers, f"found {len(states)} KV states"
        return states

    def get_kv(self, row: int = 0, dtype=np.float32, keys_only: bool = False) -> KV:
        """Per layer (K, V) of one batch row, each [kv_heads, kv_len, head_dim] (K post-RoPE)."""
        st = self._states()
        out = []
        for i in range(self.n_layers):
            k = st[(i, "key")].state.data[row].astype(dtype, copy=True)
            v = None if keys_only else st[(i, "value")].state.data[row].astype(dtype, copy=True)
            out.append((k, v))
        return out

    def layer_keys(self, layer: int, row: int = 0, dtype=np.float32) -> np.ndarray:
        """Keys of one layer [kv_heads, kv_len, head_dim] (post-RoPE); reads only that layer's state."""
        for st in self.req.query_state():
            m = _STATE_RE.search(st.name)
            if m and int(m.group(1)) == layer and m.group(2) == "key":
                return st.state.data[row].astype(dtype, copy=True)
        raise KeyError(layer)

    def set_kv(self, kv: KV, batch: int = 1, positions_next: np.ndarray | int | None = None):
        """Write the same KV (per layer [kv_heads, len, head_dim]) into `batch` rows; clears all eviction masks.

        fp16 host tensors are accepted (half the host memory). Do NOT call reset() before this: on the GPU plugin
        (2026.4.1) set_state() after reset_state() is ignored by the next inference although the state reads back
        correctly (verified: next-token logits diff 13.6 on Qwen3-4B; CPU is fine)."""
        lengths = {k.shape[1] for k, _ in kv} | {v.shape[1] for _, v in kv}
        assert len(lengths) == 1, f"all layers must have the same KV length (mask is shared), got {lengths}"
        st = self._states()
        for i, (k, v) in enumerate(kv):
            for name, arr in (("key", k), ("value", v)):
                a = np.ascontiguousarray(np.broadcast_to(arr.astype(np.float16)[None], (batch, *arr.shape)))
                st[(i, name)].state = ov.Tensor(a)
        self._clear(batch, lengths.pop())
        if positions_next is not None:
            self.next_pos = np.broadcast_to(np.asarray(positions_next, np.int64), (batch,)).copy()

    # ---- RestoreKV --------------------------------------------------------------------------------
    @property
    def n_restore(self) -> int:
        return 0 if self.ckpt is None else self.ckpt.embeddings.shape[0]

    def restore_pass(self) -> None:
        """Run the n restore embeddings over the full cache with LoRA on (batch 1); their K/V are appended to the
        state at positions next_pos..next_pos+n-1 (= T..T+n-1 right after prefill)."""
        assert self.ckpt is not None and self.batch == 1
        self.forward(self.ckpt.embeddings[None].astype(np.float32), lora=1.0)

    # ---- generation -------------------------------------------------------------------------------
    def greedy(self, first_logits: np.ndarray, max_new: int, stop_ids: set[int] = frozenset()) -> list[int]:
        """Batch-1 greedy decode."""
        return self._greedy_batch(first_logits.reshape(1, -1), max_new, stop_ids)[0]

    def _greedy_batch(self, logits: np.ndarray, max_new: int, stop_ids: set[int]) -> list[list[int]]:
        B = logits.shape[0]
        outs: list[list[int]] = [[] for _ in range(B)]
        done = np.zeros(B, bool)
        pad = next(iter(stop_ids)) if stop_ids else 0
        for _ in range(max_new):
            toks = logits.argmax(-1)
            for r in range(B):
                if not done[r]:
                    outs[r].append(int(toks[r]))
                    done[r] = int(toks[r]) in stop_ids
            if done.all():
                break
            feed = np.where(done, pad, toks).astype(np.int64)[:, None]
            logits = self.forward(self.embed(feed)).logits
        return outs

    def decode_variants(self, ctx_kv: KV, variants: list[Variant], question_ids: np.ndarray, max_new: int,
                        stop_ids: set[int], restore_kv: KV | None = None) -> list[list[int]]:
        """Decode the same question under several eviction variants in one batch.

        ctx_kv: full context KV (T entries per layer). restore_kv: the n restore entries per layer (from the
        LoRA pass over the full cache), appended to every row; rows with restore=False mask them out, so they
        behave exactly as if the entries did not exist. Question positions start at T (+n for restore rows)."""
        T = ctx_kv[0][0].shape[1]
        n_r = 0 if restore_kv is None else restore_kv[0][0].shape[1]
        assert all(not v.restore for v in variants) or n_r > 0, "restore variant without restore_kv"
        kv = ctx_kv if n_r == 0 else [(np.concatenate([k, rk], 1), np.concatenate([v, rv], 1))
                                      for (k, v), (rk, rv) in zip(ctx_kv, restore_kv)]
        B = len(variants)
        self.set_kv(kv, batch=B, positions_next=np.array([T + (n_r if v.restore else 0) for v in variants]))
        evict = np.zeros((B, self.n_layers, self.n_kv_heads, T + n_r), bool)
        for r, v in enumerate(variants):
            if v.evict_ctx is not None:
                evict[r, :, :, :T] = v.evict_ctx
            if not v.restore and n_r:
                evict[r, :, :, T:] = True
        self.mask_entries(evict)
        logits = self.forward(self.embed(np.repeat(np.asarray(question_ids)[None], B, 0))).logits
        return self._greedy_batch(logits, max_new, stop_ids)
