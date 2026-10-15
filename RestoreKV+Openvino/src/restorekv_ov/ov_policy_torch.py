"""OpenVINO GenAI's cache-eviction policy in PyTorch, for adapter training.

Port of the verified numpy implementation used by the evaluator (`scoring.ov_policy_scores`, `eviction.ov_select`), which
itself follows openvino.genai `src/cpp/src/continuous_batching/cache/cache_eviction.cpp` and the PagedAttention
`scores` output (attention weights summed over all query heads):

  1. Score: softmax attention of the last `window` (SnapKV, default 8) prompt queries over all keys, summed over all
     query heads and the window rows -> one score per (layer, token).
  2. Forward-looking max-pool over `max_pool` (7) tokens, start area excluded; NORM_SUM divides by occurrence
     counters (window for older tokens, window..1 for the last `window` tokens).
  3. Selection per layer, whole blocks, the same for all KV heads: start area and recent area (counted back from the
     last FULL block, trailing partial block kept) are never evicted; the evictable area keeps
     (max_cache_size - start - recent - n_restore) tokens rounded down to whole blocks, max_cache_size = keep * T
     rounded down to a block multiple. Blocks are aligned to absolute position 0 of the prompt.

Self-contained (torch only) so it can be dropped into the official RestoreKV training code.
"""
from __future__ import annotations

import torch


def ov_window_scores(q_window: torch.Tensor, keys: torch.Tensor, q_abs_start: int, per_head: bool = False) -> torch.Tensor:
    """Attention scores of the window queries, summed over query heads and rows.

    q_window: [Hq, W, D] post-RoPE queries of absolute positions q_abs_start..q_abs_start+W-1.
    keys: [Hkv, K, D] post-RoPE keys of absolute positions 0..K-1 (K >= q_abs_start + W).
    Returns [K] float32 (keys after the last query get 0); per_head=True sums only over each KV head's GQA group of
    query heads and returns [Hkv, K] (per-head eviction)."""
    hq, w, d = q_window.shape
    hkv, k_len, _ = keys.shape
    group = hq // hkv
    k_rep = keys.repeat_interleave(group, dim=0).float()                     # HF repeat_kv order
    logits = torch.matmul(q_window.float(), k_rep.transpose(1, 2)) / d ** 0.5  # [Hq, W, K]
    rows = q_abs_start + torch.arange(w, device=keys.device)[:, None]
    cols = torch.arange(k_len, device=keys.device)[None, :]
    logits = logits.masked_fill(cols > rows, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    if per_head:
        return p.reshape(hkv, group, w, k_len).sum(dim=(1, 2))
    return p.sum(dim=(0, 1))


def ov_postprocess(scores: torch.Tensor, window: int, start_size: int = 32, max_pool: int = 7,
                   norm_sum: bool = True) -> torch.Tensor:
    """scores [L, T] (from ov_window_scores per layer, T = prompt length) -> per-token eviction score [L, T].
    Start-area tokens get +inf (never evicted, never scored in OpenVINO)."""
    L, T = scores.shape
    tracked = scores[:, start_size:].float()
    n = tracked.shape[1]
    padded = torch.nn.functional.pad(tracked, (0, max_pool - 1), value=float("-inf"))
    pooled = padded.unfold(1, max_pool, 1).amax(-1)[:, :n]                  # dst[i] = max(src[i : i+max_pool])
    if norm_sum:
        counter = torch.full((n,), float(window), device=scores.device)
        w = min(window, n)
        counter[n - w:] = torch.arange(w, 0, -1, device=scores.device, dtype=torch.float32)
        pooled = pooled / counter
    out = torch.full((L, T), float("inf"), device=scores.device)
    out[:, start_size:] = pooled
    return out


def ov_ph_block_keep(scores: torch.Tensor, keep_frac: float, n_restore: int = 0, block: int = 16, start_size: int = 32,
                     recent_size: int = 32) -> torch.Tensor:
    """Per-head block eviction: scores [L, H, T] (per-head ov_window_scores + ov_postprocess) -> keep [L, H, T].
    OpenVINO's block rule per (layer, KV head); same number of kept blocks for every head.
    Mirrors restorekv_ov.eviction.ov_ph_select."""
    L, H, T = scores.shape
    return ov_block_keep(scores.reshape(L * H, T), keep_frac, n_restore, block, start_size, recent_size).reshape(L, H, T)


def ov_block_keep(scores: torch.Tensor, keep_frac: float, n_restore: int = 0, block: int = 16, start_size: int = 32,
                  recent_size: int = 32) -> torch.Tensor:
    """scores [L, T] -> keep mask [L, T] (bool), identical for all KV heads.

    Mirrors restorekv_ov.eviction.ov_select (evaluator) bit-for-bit, including ties (stable sort by score)."""
    L, T = scores.shape
    keep = torch.ones((L, T), dtype=torch.bool, device=scores.device)
    n_full = T // block
    first, last = start_size // block, n_full - recent_size // block
    if last <= first:
        return keep
    max_cache = int(keep_frac * T) // block * block
    keep_blocks = max(0, (max_cache - start_size - recent_size - n_restore) // block)
    blocks = scores[:, first * block:last * block].float().reshape(L, -1, block).sum(-1)     # [L, nb]
    n_evict = blocks.shape[1] - keep_blocks
    if n_evict <= 0:
        return keep
    order = torch.sort(blocks, dim=1, stable=True).indices[:, :n_evict]                     # lowest first
    evict_blocks = torch.zeros_like(blocks, dtype=torch.bool).scatter_(1, order, True)
    keep[:, first * block:last * block] = ~evict_blocks.repeat_interleave(block, dim=1)
    return keep


ARKV_EPS = torch.finfo(torch.float32).eps   # kernel EPSILON = std::numeric_limits<float>::epsilon()


def arkv_block_keep(raw_scores: torch.Tensor, keys: list[torch.Tensor], keep_frac: float, n_restore: int = 0,
                    block: int = 16, start_size: int = 32, recent_size: int = 32,
                    attention_mass: float = 0.9) -> torch.Tensor:
    """OpenVINO AggregationMode::ADAPTIVE_RKV one-shot eviction -> keep mask [L, T] (same for all KV heads).

    raw_scores [L, T]: SnapKV-window attention summed over heads (ov_window_scores), NOT max-pooled / NORM_SUM.
    keys: per layer [Hkv, T, D] post-RoPE. Mirrors restorekv_ov.eviction.arkv_select (numpy evaluator)."""
    L, T = raw_scores.shape
    dev = raw_scores.device
    keep = torch.ones((L, T), dtype=torch.bool, device=dev)
    n_full = T // block
    first, last = start_size // block, n_full - recent_size // block
    if last <= first:
        return keep
    nb = last - first
    E = nb * block
    max_cache = int(keep_frac * T) // block * block
    max_keep = max(0, (max_cache - start_size - recent_size - n_restore) // block)
    for layer in range(L):
        bs = raw_scores[layer, first * block:last * block].double().reshape(nb, block).sum(-1)
        order = torch.sort(-bs, stable=True).indices.tolist()
        expected, acc, kept = float(bs.sum()) * attention_mass, 0.0, []
        for b in order:
            if not (acc < expected and len(kept) < max_keep):
                break
            acc += float(bs[b])
            kept.append(b)
        kept_set = set(kept)
        div_set = torch.tensor([b for b in range(nb) if b not in kept_set], dtype=torch.long, device=dev)
        left = max_keep - len(kept)
        chosen = torch.zeros(nb, dtype=torch.bool, device=dev)
        chosen[kept] = True
        if left > 0 and len(div_set):
            k = keys[layer][:, first * block:last * block].float()
            kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + ARKV_EPS)
            A = torch.zeros((E, E), dtype=torch.float32, device=dev)
            for h in range(k.shape[0]):
                s = kn[h] @ kn[h].T
                s.fill_diagonal_(0.0)
                s = torch.where(s < s.mean(dim=1, keepdim=True), torch.zeros_like(s), s)
                A += s
            A /= k.shape[0]
            D = -A.reshape(nb, block, E).sum(1)
            cols = (div_set[:, None] * block + torch.arange(block, device=dev)[None]).reshape(-1)
            filtered = D[:, cols].sum(1) / (len(div_set) * block)
            top = div_set[torch.sort(-filtered[div_set], stable=True).indices[:left]]
            chosen[top] = True
        evict_blocks = ~chosen
        keep[layer, first * block:last * block] = ~evict_blocks.repeat_interleave(block)
    return keep
