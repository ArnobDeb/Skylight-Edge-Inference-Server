"""add per-head block eviction (`--level ov-headblock`) to the RestoreKV training code.

"Finer eviction": OpenVINO's block rule (SnapKV window 8, max-pool 7, NORM_SUM, whole --ov-block-token blocks,
start/recent areas, restore slots out of the block budget) applied per (layer, KV head) instead of per layer. The
scores are the window attention summed over each KV head's GQA group of query heads (not over all heads), so heads
keep different blocks; every head keeps the same NUMBER of blocks (equal-length per-head block tables, the layout a
C++ port would use). attention/ov_policy.py::ov_ph_block_keep == src/restorekv_ov/ov_policy_torch.py, verified
identical to the numpy evaluator (eviction.ov_ph_select, evaluator method `ov_ph`) by test_ov_policy_torch.py.

Applies on top of apply_ov_policy_patch.py and apply_ov_arkv_patch.py:
    python apply_ov_headblock_patch.py /path/to/RestoreKV   (copies ov_policy.py again: it now has the per-head code)
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
    if '"ov-headblock"' in tr.read_text():
        print("already patched"); return
    assert '"ov-arkv"' in tr.read_text(), "apply apply_ov_policy_patch.py and apply_ov_arkv_patch.py first"
    shutil.copy(OV_POLICY_SRC if OV_POLICY_SRC.exists() else HERE / "ov_policy_torch.py", repo / "attention/ov_policy.py")

    # --- train script: CLI, level mapping, per-head score buffers + post-processing, budget matching ------------
    replace_once(tr, '''"ov-block", "ov-arkv"], default="pair")''', '''"ov-block", "ov-arkv", "ov-headblock"], default="pair")''')
    replace_once(tr, '''    if args.level in ("ov-block", "ov-arkv"):
        # OpenVINO layout''', '''    if args.level in ("ov-block", "ov-arkv", "ov-headblock"):
        # OpenVINO layout''')
    replace_once(tr, '''    elif args.level in ("ov-block", "ov-arkv"):
        args.score_method = "ov_h2o"''', '''    elif args.level in ("ov-block", "ov-arkv", "ov-headblock"):
        args.score_method = "ov_h2o"''')
    replace_once(tr, '''            kv.ov_raw = [torch.zeros(full_len, dtype=torch.float32, device=kv.device) for _ in range(kv.n_layers)]''',
                 '''            kv.ov_per_head = args.level == "ov-headblock"
            shape = (kv.n_heads_kv, full_len) if kv.ov_per_head else (full_len,)
            kv.ov_raw = [torch.zeros(shape, dtype=torch.float32, device=kv.device) for _ in range(kv.n_layers)]''')
    replace_once(tr, '''            else:
                kv.ov_policy = "block"
                kv.ov_scores = ov_postprocess(torch.stack(kv.ov_raw), window=args.ov_window, start_size=args.ov_start)''',
                 '''            elif args.level == "ov-headblock":
                kv.ov_policy = "headblock"
                raw = torch.stack(kv.ov_raw)                                              # [L, Hkv, T]
                L_, H_, T_ = raw.shape
                kv.ov_scores = ov_postprocess(raw.reshape(L_ * H_, T_), window=args.ov_window,
                                              start_size=args.ov_start).reshape(L_, H_, T_)
            else:
                kv.ov_policy = "block"
                kv.ov_scores = ov_postprocess(torch.stack(kv.ov_raw), window=args.ov_window, start_size=args.ov_start)''')
    replace_once(tr, '''            kv.score = [s[None, None, sys_len:].expand(1, kv.n_heads_kv, -1).to(kv.dtype)
                        for s in kv.ov_scores.clamp(max=torch.finfo(kv.dtype).max)]''',
                 '''            kv.score = [(s[None, :, sys_len:] if s.dim() == 2 else s[None, None, sys_len:].expand(1, kv.n_heads_kv, -1))
                        .to(kv.dtype) for s in kv.ov_scores.clamp(max=torch.finfo(kv.dtype).max)]''')

    # --- score.py: per-head accumulation -----------------------------------------------------------------------
    score = repo / "attention/score.py"
    replace_once(score, '''        s = ov_window_scores(rows, key_states[0], lo)
        self.ov_raw[layer_idx][:k_total] += s''', '''        if getattr(self, "ov_per_head", False):
            s = ov_window_scores(rows, key_states[0], lo, per_head=True)      # [Hkv, k_total]
            self.ov_raw[layer_idx][:, :k_total] += s
        else:
            s = ov_window_scores(rows, key_states[0], lo)
            self.ov_raw[layer_idx][:k_total] += s''')

    # --- kvcache.py: route ov-headblock to ov_ph_block_keep ----------------------------------------------------
    kvc = repo / "attention/kvcache.py"
    s = kvc.read_text()
    assert s.count('        if level in ("ov-block", "ov-arkv"):\n') == 3
    kvc.write_text(s.replace('        if level in ("ov-block", "ov-arkv"):\n',
                             '        if level in ("ov-block", "ov-arkv", "ov-headblock"):\n'))
    replace_once(kvc, '''    from attention.ov_policy import arkv_block_keep, ov_block_keep
    if getattr(self, "ov_policy", "block") == "arkv":''', '''    from attention.ov_policy import arkv_block_keep, ov_block_keep, ov_ph_block_keep
    if getattr(self, "ov_policy", "block") == "headblock":
        keep = ov_ph_block_keep(self.ov_scores, ratio, n_restore=getattr(self, "ov_n_restore", 0),
                                block=self.ov_block, start_size=self.ov_start,
                                recent_size=self.ov_recent)                         # [L, Hkv, sys + ctx]
        return keep[:, None, :, self.start_idx:self.end_idx].contiguous()          # [L, 1, Hkv, ctx]
    if getattr(self, "ov_policy", "block") == "arkv":''')
    print("patched", repo)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
