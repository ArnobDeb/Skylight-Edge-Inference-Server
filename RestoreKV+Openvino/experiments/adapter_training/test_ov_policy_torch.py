"""The PyTorch port of OpenVINO's eviction policy (used for training) must match the numpy
implementation (used for evaluation) on the same queries/keys: same scores and the same kept blocks.

Two policies:
  ov     OpenVINO H2O / NORM_SUM: head-summed scores, blocks shared by all KV heads (what OpenVINO does today)
  ov_ph  per-head blocks ("finer eviction"): scores summed per GQA group, blocks chosen per (layer, KV head)

Evidence: one CSV row per (case, layer[, head]) with the kept block IDs from both implementations, written to
results/adapter_training/<date>_policy_port_tests/{ov,ov_ph}_cases.csv. Block b = tokens [b*block, (b+1)*block), aligned to
position 0; a block counts as kept if any of its tokens is kept (eviction is whole-block, so it is all or nothing).

    conda activate arnob
    env -u PYTHONPATH python experiments/adapter_training/test_ov_policy_torch.py
"""
from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov.eviction import ov_ph_select, ov_select  # noqa: E402
from restorekv_ov.ov_policy_torch import ov_block_keep, ov_ph_block_keep, ov_postprocess, ov_window_scores  # noqa: E402
from restorekv_ov.scoring import ov_policy_scores  # noqa: E402

OUT = ROOT / "results/adapter_training" / f"{date.today()}_policy_port_tests"
FIELDS = ["case", "policy", "T", "keep_frac", "n_restore", "block", "start", "recent", "layer", "head",
          "n_blocks", "kept_numpy", "kept_torch", "n_kept", "agree"]


class FakeEngine:
    """Just enough of OVRestoreEngine for ov_policy_scores."""

    def __init__(self, keys):
        self.keys = keys                         # per layer [Hkv, T, D]
        self.kv_len = keys[0].shape[1]
        self.n_kv_heads, self.head_dim = keys[0].shape[0], keys[0].shape[2]
        self.n_q_heads = 32
        self.n_layers = len(keys)

    def get_kv(self, keys_only=False, **_):
        return [(k, None) for k in self.keys]


def kept_blocks(keep_row: np.ndarray, block: int) -> list[int]:
    n = -(-len(keep_row) // block)
    return [b for b in range(n) if keep_row[b * block:(b + 1) * block].any()]


def run_case(case: int, T: int, keep: float, n_restore: int, block: int, start: int, recent: int, per_head: bool,
             rows: list[dict]) -> tuple[float, bool]:
    rng = np.random.default_rng(case)
    L, Hkv, Hq, D, W = 4, 8, 32, 128, 8
    keys = [rng.standard_normal((Hkv, T, D)).astype(np.float32) for _ in range(L)]
    qwin = [rng.standard_normal((1, Hq, W, D)).astype(np.float32) for _ in range(L)]

    ref_scores = ov_policy_scores(FakeEngine(keys), qwin, start_size=start, per_head=per_head)   # [L, T] / [L, H, T]
    raw = torch.stack([ov_window_scores(torch.from_numpy(q[0]), torch.from_numpy(k), T - W, per_head=per_head)
                       for q, k in zip(qwin, keys)])
    if per_head:
        ref_keep = ~ov_ph_select(ref_scores, keep, n_restore, block=block, start_size=start, recent_size=recent)
        t_scores = ov_postprocess(raw.reshape(L * Hkv, T), window=W, start_size=start).reshape(L, Hkv, T)
        t_keep = ov_ph_block_keep(t_scores, keep, n_restore, block=block, start_size=start, recent_size=recent)
    else:
        ref_keep = ~ov_select(ref_scores, keep, n_restore, n_kv_heads=1, block=block,
                              start_size=start, recent_size=recent)[:, 0]                       # [L, T]
        t_scores = ov_postprocess(raw, window=W, start_size=start)
        t_keep = ov_block_keep(t_scores, keep, n_restore, block=block, start_size=start, recent_size=recent)
    t_keep = t_keep.numpy()

    finite = np.isfinite(ref_scores)
    t_np = t_scores.numpy()
    rel = float(np.max(np.abs(t_np[finite] - ref_scores[finite]) / (np.abs(ref_scores[finite]) + 1e-12)))
    same = bool((t_keep == ref_keep).all())

    heads = range(Hkv) if per_head else [None]
    for layer in range(L):
        for h in heads:
            r_ref = ref_keep[layer] if h is None else ref_keep[layer, h]
            r_t = t_keep[layer] if h is None else t_keep[layer, h]
            kb_ref, kb_t = kept_blocks(r_ref, block), kept_blocks(r_t, block)
            rows.append({"case": case, "policy": "ov_ph" if per_head else "ov", "T": T, "keep_frac": keep,
                         "n_restore": n_restore, "block": block, "start": start, "recent": recent, "layer": layer,
                         "head": "all" if h is None else h, "n_blocks": -(-T // block),
                         "kept_numpy": " ".join(map(str, kb_ref)), "kept_torch": " ".join(map(str, kb_t)),
                         "n_kept": len(kb_ref), "agree": bool((r_ref == r_t).all())})
    return rel, same


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for per_head in (False, True):
        name = "ov_ph" if per_head else "ov"
        rows: list[dict] = []
        worst, n_same, n = 0.0, 0, 0
        for T in (700, 2049, 4096, 5127):
            for keep in (0.025, 0.05, 0.1, 0.25):
                for n_restore in (0, 8):
                    for block, start, recent in ((16, 32, 32), (32, 32, 128)):
                        rel, same = run_case(n, T, keep, n_restore, block, start, recent, per_head, rows)
                        worst, n_same, n = max(worst, rel), n_same + same, n + 1
                        if not same:
                            print(f"MISMATCH {name} T={T} keep={keep} n_restore={n_restore} block={block}")
        with open(OUT / f"{name}_cases.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        passed = n_same == n and worst < 1e-4
        ok &= passed
        print(f"{name}: {n} cases ({len(rows)} layer/head rows), max relative score diff {worst:.2e}, "
              f"identical kept blocks: {n_same}/{n} -> {'PASS' if passed else 'FAIL'}  [{OUT / (name + '_cases.csv')}]")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
