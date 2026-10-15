"""Eviction scorers for the OpenVINO prototype.

kvzip_scores   - KVzip (arXiv 2505.23416), as implemented in kvpress KVzipPress (artifacts/ref/kvpress): replay
                 the context chunk-by-chunk after a "repeat" prompt, score every (layer, kv-head, token) by the max
                 softmax attention it receives from the replay queries (max over GQA group and query rows).
                 The released RestoreKV adapters were trained against this scorer ("score_method": "kvzip").
ov_policy_scores - OpenVINO GenAI's eviction score for the first eviction after prefill
                 (src/cpp/src/continuous_batching/cache/cache_eviction.cpp + CPU PA kernel): softmax of the last
                 `snapkv_window` prompt queries, summed over all query heads and window rows, max-pooled over 7
                 tokens, divided by NORM_SUM occurrence counters. One score per (layer, token).
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from openvino import opset13 as ops

from .engine import OVRestoreEngine


class _AttnMaxKernel:
    """GPU kernel: out[h, n] = max_{g, q} softmax_n(q[h, g, q] . k[h, n] * scale + bias[q, n])."""

    def __init__(self, core: ov.Core, device: str, head_dim: int):
        q = ops.parameter(ov.PartialShape([-1, -1, -1, head_dim]), ov.Type.f32, name="q")   # [Hkv, G, Q, D]
        k = ops.parameter(ov.PartialShape([-1, -1, head_dim]), ov.Type.f32, name="k")       # [Hkv, N, D]
        bias = ops.parameter(ov.PartialShape([-1, -1]), ov.Type.f32, name="bias")            # [Q, N]
        logits = ops.matmul(q, ops.unsqueeze(k, ops.constant(np.int64(1))), False, True)    # [Hkv, G, Q, N]
        logits = ops.add(ops.multiply(logits, ops.constant(np.float32(1 / np.sqrt(head_dim)))), bias)
        probs = ops.softmax(logits, -1)
        out = ops.reduce_max(probs, ops.constant(np.array([1, 2], dtype=np.int64)), False)  # [Hkv, N]
        model = ov.Model([out], [q, k, bias], "attn_max")
        self.compiled = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f32"})

    def __call__(self, q: np.ndarray, k: np.ndarray, bias: np.ndarray) -> np.ndarray:
        # one KV head at a time: the [G, Q, N] logits of all 8 heads at once take >1 GB of (shared) GPU memory
        return np.concatenate([self.compiled({"q": q[h:h + 1], "k": k[h:h + 1], "bias": bias})[0]
                               for h in range(q.shape[0])], axis=0)


def _causal_tail_bias(q_len: int, n_keys: int, q_offset: int = 0, tail_len: int | None = None) -> np.ndarray:
    """Bias [q_len, n_keys]: keys before the trailing `tail_len` block are visible to all rows; inside the tail,
    row i (global index q_offset + i of the tail sequence) sees tail keys 0..q_offset+i (causal)."""
    tail_len = q_len if tail_len is None else tail_len
    bias = np.zeros((q_len, n_keys), np.float32)
    rows = q_offset + np.arange(q_len)[:, None]
    cols = np.arange(tail_len)[None, :]
    bias[:, n_keys - tail_len:][cols > rows] = np.finfo(np.float32).min / 4
    return bias


def kvzip_scores(eng: OVRestoreEngine, ctx_ids: np.ndarray, prefix_len: int, encode, suffix_ids: np.ndarray,
                 chunk_size: int = 2048, n_sink: int = 4, prev_postfix: int = 8, kernel: _AttnMaxKernel | None = None,
                 core: ov.Core | None = None, device: str = "GPU", sub_rows: int = 512) -> np.ndarray:
    """KVzip scores [n_layers, n_kv_heads, T] for a context already prefilled into `eng` (batch 1, kv_len == T).
    The cache is restored to exactly T entries afterwards. `encode(str) -> np.ndarray[int]` without special tokens.

    Each replay sequence is fed in sub-chunks of `sub_rows` rows. This is exact (causal attention: each row's softmax
    only sees keys up to itself) and bounds the query-tap memory (36 layers x 32 heads x rows x 128 x 4 B)."""
    T = eng.kv_len
    assert len(ctx_ids) == T and eng.batch == 1
    kernel = kernel or _AttnMaxKernel(core or ov.Core(), device, eng.head_dim)
    group = eng.n_q_heads // eng.n_kv_heads
    scores = np.zeros((eng.n_layers, eng.n_kv_heads, T), np.float32)
    body = ctx_ids[prefix_len:]
    chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)]
    start = prefix_len
    for i, a_ids in enumerate(chunks):
        if i == 0:
            q_ids = encode("\n\nRepeat the previous context exactly.")
        else:
            q_ids = np.concatenate([encode("\n\nRepeat the part of the previous context exactly, starting with"),
                                    chunks[i - 1][-prev_postfix:]])
        repeat = np.concatenate([q_ids, suffix_ids, a_ids]).astype(np.int64)
        end = start + len(a_ids)
        sink = min(n_sink, start)
        n_ctx = end - start
        for s0 in range(0, len(repeat), sub_rows):
            part = repeat[s0:s0 + sub_rows]
            out = eng.forward(eng.embed(part), q_rows=len(part))   # positions continue at T + s0, full cache visible
            keys_all = eng.get_kv(keys_only=True)
            n_tail = s0 + len(part)                                 # replay keys present so far
            bias = _causal_tail_bias(len(part), sink + n_ctx + n_tail, q_offset=s0, tail_len=n_tail)
            for layer, (k, _) in enumerate(keys_all):
                keys = np.concatenate([k[:, :sink], k[:, start:end], k[:, T:T + n_tail]], axis=1)
                q = out.queries[layer][0].reshape(eng.n_kv_heads, group, len(part), eng.head_dim)
                sc = kernel(q, keys, bias)[:, sink:sink + n_ctx]
                np.maximum(scores[layer, :, start:end], sc, out=scores[layer, :, start:end])
        kv = eng.get_kv(dtype=np.float16)
        eng.set_kv([(k[:, :T], v[:, :T]) for k, v in kv])        # drop the replay tokens (no reset(): GPU bug)
        start = end
    return scores


def ov_policy_scores(eng: OVRestoreEngine, window_queries: list[np.ndarray], start_size: int = 32,
                     max_pool: int = 7, norm_sum: bool = True, per_head: bool = False) -> np.ndarray:
    """OpenVINO's score for the one-shot eviction after prefill, per (layer, token) [n_layers, T].
    per_head=True ("finer eviction", not in OpenVINO today): the attention is summed only over the query heads
    of each KV head's GQA group, giving [n_layers, n_kv_heads, T] with the same max-pool / NORM_SUM per head.
    window_queries: per layer [1, n_q_heads, W, head_dim], the last W prompt rows (prefill with q_rows_last=W).
    Tokens in the start area get +inf (never evicted, never scored in OpenVINO).

    Note: for this one-shot eviction SUM and NORM_SUM rank tokens identically outside the last W positions (the
    NORM_SUM counter is the constant W there, and the last W tokens sit in the recent area). They only diverge
    when eviction repeats during long generation."""
    T = eng.kv_len
    group = eng.n_q_heads // eng.n_kv_heads
    out = np.zeros((eng.n_layers, eng.n_kv_heads, T) if per_head else (eng.n_layers, T), np.float32)
    all_kv = None if hasattr(eng, "layer_keys") else eng.get_kv(keys_only=True)
    for layer in range(eng.n_layers):
        k = eng.layer_keys(layer)[:, :T] if all_kv is None else all_kv[layer][0]   # [Hkv, T, D], one layer at a time
        q = window_queries[layer][0]                        # [Hq, W, D]
        W = q.shape[1]
        # GQA without materializing repeat_kv: query head h = kv_head * group + g uses key head kv_head
        qg = q.reshape(eng.n_kv_heads, group * W, -1)
        logits = (np.matmul(qg, k.transpose(0, 2, 1)) / np.sqrt(eng.head_dim)).reshape(eng.n_q_heads, W, T)
        logits += _causal_tail_bias(W, T, tail_len=W)[None]
        logits -= logits.max(-1, keepdims=True)
        p = np.exp(logits); p /= p.sum(-1, keepdims=True)
        if per_head:
            s = p.reshape(eng.n_kv_heads, group, W, T).sum(axis=(1, 2))   # [Hkv, T]: sum over the group's q-heads
        else:
            s = p.sum(axis=(0, 1))[None]                    # sum over heads and window rows (kernel attn_reduce + SnapKV)
        tracked = s[:, start_size:]                         # start area is not tracked
        n = tracked.shape[1]
        padded = np.concatenate([tracked, np.full((tracked.shape[0], max_pool - 1), -np.inf, tracked.dtype)], axis=1)
        win = np.lib.stride_tricks.sliding_window_view(padded, max_pool, axis=1)
        pooled = win.max(axis=-1)[:, :n].astype(np.float32)  # forward-looking max-pool, as cache_eviction.cpp _max_pool
        if norm_sum:
            counter = np.full(n, W, np.float32)
            counter[-W:] = np.arange(W, 0, -1)              # occurrence counters after a SnapKV window of W
            pooled = pooled / counter
        res = np.empty((pooled.shape[0], T), np.float32)
        res[:, :start_size] = np.inf
        res[:, start_size:] = pooled
        out[layer] = res if per_head else res[0]
    return out
