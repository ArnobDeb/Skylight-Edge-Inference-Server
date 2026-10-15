"""The PyTorch port of OpenVINO's ADAPTIVE_RKV eviction (training) must keep exactly the same blocks as the
numpy evaluator, on the same raw scores and keys.

Keys are drawn from a few clusters so the diversity stage actually decides which blocks survive.
Evidence: one CSV row per (case, layer) with the kept block IDs from both implementations, written to
results/adapter_training/<date>_policy_port_tests/arkv_cases.csv (block b = tokens [b*block, (b+1)*block), aligned to 0).

    conda activate arnob
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python experiments/adapter_training/test_arkv_torch.py
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
from restorekv_ov.eviction import arkv_select  # noqa: E402
from restorekv_ov.ov_policy_torch import arkv_block_keep, ov_window_scores  # noqa: E402

OUT = ROOT / "results/adapter_training" / f"{date.today()}_policy_port_tests"


def kept_blocks(keep_row: np.ndarray, block: int) -> list[int]:
    n = -(-len(keep_row) // block)
    return [b for b in range(n) if keep_row[b * block:(b + 1) * block].any()]


def make_keys(rng, L, H, T, D, n_clusters=6):
    centers = rng.standard_normal((L, H, n_clusters, D)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=T)
    return [(centers[l][:, assign] + 0.3 * rng.standard_normal((H, T, D))).astype(np.float32) for l in range(L)]


def main():
    L, H, Hq, D, W = 3, 8, 32, 128, 8
    n_cases, n_same, n_diverse_used = 0, 0, 0
    rows = []
    for seed, T in enumerate((700, 1500, 2600)):
        rng = np.random.default_rng(seed)
        keys = make_keys(rng, L, H, T, D)
        q = [rng.standard_normal((Hq, W, D)).astype(np.float32) for _ in range(L)]
        raw_t = torch.stack([ov_window_scores(torch.from_numpy(qq), torch.from_numpy(k), T - W) for qq, k in zip(q, keys)])
        raw_np = raw_t.numpy()
        for keep in (0.05, 0.1, 0.25):
            for n_restore in (0, 8):
                for block, start, recent in ((16, 32, 32), (32, 32, 128)):
                    for mass in (0.9, 0.5):
                        ref = ~arkv_select(raw_np, keys, keep, n_restore, n_kv_heads=1, block=block,
                                           start_size=start, recent_size=recent, attention_mass=mass)[:, 0]
                        got = arkv_block_keep(raw_t, [torch.from_numpy(k) for k in keys], keep, n_restore,
                                              block=block, start_size=start, recent_size=recent,
                                              attention_mass=mass).numpy()
                        for layer in range(L):
                            kr, kt = kept_blocks(ref[layer], block), kept_blocks(got[layer], block)
                            rows.append({"case": n_cases, "policy": "arkv", "T": T, "keep_frac": keep,
                                         "n_restore": n_restore, "block": block, "start": start, "recent": recent,
                                         "attention_mass": mass, "layer": layer, "n_blocks": -(-T // block),
                                         "kept_numpy": " ".join(map(str, kr)), "kept_torch": " ".join(map(str, kt)),
                                         "n_kept": len(kr), "agree": bool((ref[layer] == got[layer]).all())})
                        n_cases += 1
                        same = bool((ref == got).all())
                        n_same += same
                        if not same:
                            print(f"MISMATCH T={T} keep={keep} n_restore={n_restore} block={block} mass={mass}: "
                                  f"{int((ref != got).sum())} tokens differ")
                        # did the diversity stage choose anything? (fewer kept blocks than attention-mass-only)
                        n_diverse_used += int(mass == 0.5)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "arkv_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"evidence: {OUT / 'arkv_cases.csv'} ({len(rows)} layer rows)")
    print(f"{n_cases} cases ({n_diverse_used} with attention_mass 0.5, where diversity fills most of the budget): "
          f"{n_same} identical")
    print("PASS" if n_same == n_cases else "FAIL")


if __name__ == "__main__":
    main()
