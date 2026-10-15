"""add OpenVINO's ADAPTIVE_RKV eviction (`--level ov-arkv`) to the RestoreKV training code.

Applies on top of apply_ov_policy_patch.py (branch ov-policy). Scoring is the same SnapKV-window head-summed attention
as ov-block (score_mode ov_h2o), but ADAPTIVE_RKV uses it raw (no max-pool, no NORM_SUM) and selects blocks with
attention-mass + key-diversity (attention/ov_policy.py::arkv_block_keep == src/restorekv_ov/ov_policy_torch.py,
verified identical to the evaluator by test_arkv_torch.py).

    python apply_ov_arkv_patch.py /path/to/RestoreKV      (copies ov_policy.py again: it now contains arkv_block_keep)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OV_POLICY_SRC = HERE.parents[1] / "src/restorekv_ov/ov_policy_torch.py"


def replace_once(path: Path, old: str, new: str):
    s = path.read_text()
    n = s.count(old)
    assert n == 1, f"{path.name}: anchor found {n} times:\n{old[:200]}"
    path.write_text(s.replace(old, new))


def main(repo: Path):
    tr = repo / "experiments/train_learnable_restore.py"
    if '"ov-arkv"' in tr.read_text():
        print("already patched"); return
    assert (repo / "attention/ov_policy.py").exists(), "apply apply_ov_policy_patch.py first"
    src = OV_POLICY_SRC if OV_POLICY_SRC.exists() else HERE / "ov_policy_torch.py"
    shutil.copy(src, repo / "attention/ov_policy.py")

    # --- train script: CLI, level mapping, raw scores for ARKV, budget matching ---------------------------------
    replace_once(tr, '''"random", "ov-block"], default="pair")''', '''"random", "ov-block", "ov-arkv"], default="pair")
    parser.add_argument("--ov-attention-mass", type=float, default=0.9,
                        help="ov-arkv: OpenVINO adaptive_rkv_config.attention_mass")''')
    replace_once(tr, '''    elif args.level == "ov-block":
        args.score_method = "ov_h2o"''', '''    elif args.level in ("ov-block", "ov-arkv"):
        args.score_method = "ov_h2o"''')
    replace_once(tr, '''            kv.ov_scores = ov_postprocess(torch.stack(kv.ov_raw), window=args.ov_window, start_size=args.ov_start)''',
                 '''            if args.level == "ov-arkv":
                # ADAPTIVE_RKV registers the first scores with max-pool window 1 and no NORM_SUM counters
                kv.ov_policy, kv.ov_attention_mass = "arkv", args.ov_attention_mass
                kv.ov_scores = ov_postprocess(torch.stack(kv.ov_raw), window=args.ov_window, start_size=args.ov_start,
                                              max_pool=1, norm_sum=False)
            else:
                kv.ov_policy = "block"
                kv.ov_scores = ov_postprocess(torch.stack(kv.ov_raw), window=args.ov_window, start_size=args.ov_start)''')
    replace_once(tr, '''    if args.level == "ov-block":
        # OpenVINO layout''', '''    if args.level in ("ov-block", "ov-arkv"):
        # OpenVINO layout''')

    # --- kvcache.py: route ov-arkv to arkv_block_keep (keys from the cache, prompt positions only) ----------------
    kvc = repo / "attention/kvcache.py"
    s = kvc.read_text()
    assert s.count('        if level == "ov-block":\n') == 3, "expected the ov-block branch in all three prune() methods"
    s = s.replace('        if level == "ov-block":\n', '        if level in ("ov-block", "ov-arkv"):\n')
    kvc.write_text(s)
    replace_once(kvc, '''    from attention.ov_policy import ov_block_keep
    keep = ov_block_keep(self.ov_scores, ratio, n_restore=getattr(self, "ov_n_restore", 0), block=self.ov_block,
                         start_size=self.ov_start, recent_size=self.ov_recent)       # [L, sys + ctx]''',
                 '''    from attention.ov_policy import arkv_block_keep, ov_block_keep
    if getattr(self, "ov_policy", "block") == "arkv":
        T = self.ov_scores.shape[1]
        keys = [self.key_cache[l][0, :, :T] for l in range(self.n_layers)]          # post-RoPE, prompt positions
        keep = arkv_block_keep(self.ov_scores, keys, ratio, n_restore=getattr(self, "ov_n_restore", 0),
                               block=self.ov_block, start_size=self.ov_start, recent_size=self.ov_recent,
                               attention_mass=self.ov_attention_mass)              # [L, sys + ctx]
    else:
        keep = ov_block_keep(self.ov_scores, ratio, n_restore=getattr(self, "ov_n_restore", 0), block=self.ov_block,
                             start_size=self.ov_start, recent_size=self.ov_recent)   # [L, sys + ctx]''')
    print("patched", repo)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
