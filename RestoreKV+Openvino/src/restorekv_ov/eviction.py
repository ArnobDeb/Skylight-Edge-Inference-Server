"""Eviction selection: score -> boolean evict mask over the T context entries (restore entries are never evicted).

Budgets are the fraction of the context KEPT (5% / 10% / 25%). With RestoreKV the n restore entries per
(layer, head) come out of the same budget (budget-matched, as in kvpress RestoreKVPress.compress_post).
"""
from __future__ import annotations

import numpy as np


def kvzip_select(scores: np.ndarray, keep_frac: float, n_restore: int = 0, n_sink: int = 4) -> np.ndarray:
    """KVzip / kvpress `compress_post` (layerwise=False): global bottom-k over all (layer, head, token) entries.
    scores [L, H, T] -> evict bool [L, H, T]. Per-head and per-layer kept counts are non-uniform."""
    L, H, T = scores.shape
    s = scores.copy()
    s[..., :n_sink] = s.max() + 1.0
    remove_frac = min(1.0, (1.0 - keep_frac) + n_restore / T)
    n_pruned = int(s.size * remove_frac)
    evict = np.zeros(s.size, bool)
    if n_pruned:
        evict[np.argpartition(s.reshape(-1), n_pruned - 1)[:n_pruned]] = True
    return evict.reshape(L, H, T)


def ov_select(scores: np.ndarray, keep_frac: float, n_restore: int = 0, n_kv_heads: int = 8, block: int = 32,
              start_size: int = 32, recent_size: int = 128) -> np.ndarray:
    """OpenVINO CacheEvictionAlgorithm, one-shot after prefill, per layer and whole blocks, same for all heads.
    scores [L, T] -> evict bool [L, n_kv_heads, T].

    Arena: start area (start_size) + evictable blocks + recent area (recent_size, counted back from the last
    full block); the tokens of a trailing partial block are kept too. The budget max_cache_size =
    keep_frac * T (rounded down to a block multiple); the evictable area keeps
    (max_cache_size - start - recent - n_restore) tokens, rounded down to whole blocks, so restore never
    exceeds the plain-eviction budget."""
    L, T = scores.shape
    n_full = T // block
    first, last = start_size // block, n_full - recent_size // block       # evictable block range [first, last)
    max_cache = int(keep_frac * T) // block * block
    keep_blocks = max(0, (max_cache - start_size - recent_size - n_restore) // block)
    evict = np.zeros((L, T), bool)
    if last <= first:
        return np.repeat(evict[:, None], n_kv_heads, axis=1)
    for layer in range(L):
        blocks = scores[layer, first * block:last * block].reshape(-1, block).sum(-1)
        n_evict = max(0, len(blocks) - keep_blocks)
        if n_evict:
            worst = np.argsort(blocks, kind="stable")[:n_evict]
            for b in worst:
                s = (first + b) * block
                evict[layer, s:s + block] = True
    return np.repeat(evict[:, None], n_kv_heads, axis=1)


def ov_ph_select(scores: np.ndarray, keep_frac: float, n_restore: int = 0, block: int = 16, start_size: int = 32,
                 recent_size: int = 32) -> np.ndarray:
    """Per-head block eviction ("finer eviction"): OpenVINO's block rule (start/recent areas, whole blocks,
    budget, restore slots) applied to every (layer, KV head) separately with per-head scores, so heads keep different
    blocks but the same NUMBER of blocks (equal-length per-head block tables). block=1 gives per-head tokens.
    scores [L, H, T] (ov_policy_scores(per_head=True)) -> evict bool [L, H, T]."""
    L, H, T = scores.shape
    return ov_select(scores.reshape(L * H, T), keep_frac, n_restore, n_kv_heads=1, block=block,
                     start_size=start_size, recent_size=recent_size).reshape(L, H, T)


def kvzip_layer_select(scores: np.ndarray, keep_frac: float, n_restore: int = 0, n_sink: int = 4) -> np.ndarray:
    """kvpress KVzip `layerwise=True`: the same number of entries per layer, bottom-k over (head, token) inside the
    layer, so heads still keep different counts. scores [L, H, T] -> evict [L, H, T]."""
    L, H, T = scores.shape
    out = np.zeros((L, H, T), bool)
    remove_frac = min(1.0, (1.0 - keep_frac) + n_restore / T)
    n_pruned = int(H * T * remove_frac)
    for layer in range(L):
        s = scores[layer].copy()
        s[:, :n_sink] = s.max() + 1.0
        if n_pruned:
            flat = np.zeros(H * T, bool)
            flat[np.argpartition(s.reshape(-1), n_pruned - 1)[:n_pruned]] = True
            out[layer] = flat.reshape(H, T)
    return out


def headshared_select(scores: np.ndarray, keep_frac: float, n_restore: int = 0, n_sink: int = 4) -> np.ndarray:
    """Per layer, token-level, ONE token set shared by all heads (OpenVINO's head sharing, without blocks).
    Token score = max over heads. Keeps keep_frac*T - n_restore tokens per layer. scores [L, H, T] -> evict [L, H, T]."""
    L, H, T = scores.shape
    tok = scores.max(axis=1).copy()
    tok[:, :n_sink] = np.inf
    n_keep = max(n_sink, int(keep_frac * T) - n_restore)
    out = np.ones((L, T), bool)
    for layer in range(L):
        out[layer, np.argpartition(-tok[layer], n_keep - 1)[:n_keep]] = False
    return np.repeat(out[:, None], H, axis=1)


def kvzip_block_select(scores: np.ndarray, keep_frac: float, n_restore: int = 0, block: int = 32,
                       start_size: int = 32, recent_size: int = 128) -> np.ndarray:
    """KVzip scores under OpenVINO's cache layout: per layer, whole blocks, heads shared, start/recent areas kept.
    Token score = max over heads; then exactly `ov_select`. scores [L, H, T] -> evict [L, H, T]."""
    return ov_select(scores.max(axis=1), keep_frac, n_restore, n_kv_heads=scores.shape[1], block=block,
                     start_size=start_size, recent_size=recent_size)


ARKV_EPS = np.float32(np.finfo(np.float32).eps)   # kernel EPSILON = std::numeric_limits<float>::epsilon()


def arkv_block_diversity(keys: np.ndarray) -> np.ndarray:
    """OpenVINO GPU kernel pa_adaptive_rkv_diversity_ref.cl for one layer.

    keys [Hkv, E, D] = keys of the E evictable tokens (whole blocks). Per head: cosine similarity of all token pairs
    (eps inside the sqrt), diagonal 0, entries below their row mean (over all E columns) set to 0; mean over heads ->
    A [E, E]. Returns D [nb, E] with D[b, j] = -sum_{i in block b} A[i, j] (block size = E // nb is applied by caller)."""
    k = keys.astype(np.float32)
    kn = k / np.sqrt((k * k).sum(-1, keepdims=True) + ARKV_EPS)
    H, E, _ = k.shape
    A = np.zeros((E, E), np.float32)
    for h in range(H):
        s = kn[h] @ kn[h].T
        np.fill_diagonal(s, 0.0)
        s[s < s.mean(axis=1, keepdims=True)] = 0.0
        A += s
    return A / H


def arkv_select(raw_scores: np.ndarray, keys: list[np.ndarray], keep_frac: float, n_restore: int = 0,
                n_kv_heads: int = 8, block: int = 16, start_size: int = 32, recent_size: int = 32,
                attention_mass: float = 0.9) -> np.ndarray:
    """OpenVINO AggregationMode::ADAPTIVE_RKV, one-shot eviction after prefill (cache_eviction.cpp:
    AdaptiveRKVBlockCalculator). raw_scores [L, T]: SnapKV-window attention summed over heads, NOT max-pooled and NOT
    NORM_SUM-normalised (ADAPTIVE_RKV registers scores with pool window 1). keys: per layer [Hkv, T, D] post-RoPE.

    1) Keep the highest-scoring evictable blocks until they hold `attention_mass` of the evictable attention or the
       block budget is reached. 2) Fill the remaining budget with the most *diverse* of the other blocks (lowest key
       similarity to the candidate set, kernel pa_adaptive_rkv_diversity_ref.cl). Everything else in the evictable
       range is evicted; start area, recent area and the trailing partial block are kept. Same for all KV heads."""
    L, T = raw_scores.shape
    n_full = T // block
    first, last = start_size // block, n_full - recent_size // block
    evict = np.zeros((L, T), bool)
    if last <= first:
        return np.repeat(evict[:, None], n_kv_heads, axis=1)
    nb = last - first
    max_cache = int(keep_frac * T) // block * block
    max_keep = max(0, (max_cache - start_size - recent_size - n_restore) // block)
    for layer in range(L):
        bs = raw_scores[layer, first * block:last * block].astype(np.float64).reshape(nb, block).sum(-1)
        order = np.argsort(-bs, kind="stable")
        expected, acc, kept = bs.sum() * attention_mass, 0.0, []
        for b in order:
            if not (acc < expected and len(kept) < max_keep):
                break
            acc += bs[b]
            kept.append(b)
        div_set = np.array(sorted(set(range(nb)) - set(kept)), dtype=np.int64)
        left = max_keep - len(kept)
        chosen = set()
        if left > 0 and len(div_set):
            A = arkv_block_diversity(keys[layer][:, first * block:last * block])
            D = -A.reshape(nb, block, nb * block).sum(1)                            # [nb, E]
            cols = (div_set[:, None] * block + np.arange(block)[None]).reshape(-1)
            filtered = D[:, cols].sum(1) / (len(div_set) * block)                   # get_filtered_block_diversity
            top = div_set[np.argsort(-filtered[div_set], kind="stable")[:left]]     # get_most_diverse_blocks
            chosen = set(top.tolist())
        for b in div_set:
            if b not in chosen:
                s = (first + b) * block
                evict[layer, s:s + block] = True
    return np.repeat(evict[:, None], n_kv_heads, axis=1)
