"""patch the official RestoreKV training code (cvsp-lab/RestoreKV @ 2dbcfa9) with OpenVINO's eviction policy.

Adds `--level ov-block`: scores with OpenVINO GenAI's policy during prefill (score_mode "ov_h2o": SnapKV window of the
last `--ov-window` prompt queries, attention summed over all query heads, max-pool 7, NORM_SUM) and evicts per layer
in whole `--ov-block`-token blocks shared by all KV heads, keeping `--ov-start` / `--ov-recent` areas
(attention/ov_policy.py == src/restorekv_ov/ov_policy_torch.py, verified identical to the evaluator by
test_ov_policy_torch.py). Budget-matched: the restore slots come out of the block budget exactly as at inference.

Run inside the repo (on a git branch):  python apply_ov_policy_patch.py /path/to/RestoreKV
Every replacement asserts its anchor text exists exactly once, so a mismatched upstream version fails loudly.
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
    if (repo / "attention/ov_policy.py").exists():
        print("already patched"); return
    shutil.copy(OV_POLICY_SRC if OV_POLICY_SRC.exists() else HERE / "ov_policy_torch.py", repo / "attention/ov_policy.py")

    # --- score.py: scoring hook during prefill -------------------------------------------------------------------
    score = repo / "attention/score.py"
    replace_once(score, '''        if self.score_mode == "h2o":
            return self._get_score_h2o(query_states, key_states, layer_idx)
''', '''        if self.score_mode == "h2o":
            return self._get_score_h2o(query_states, key_states, layer_idx)
        if self.score_mode == "ov_h2o":
            return self._get_score_ov(query_states, key_states, layer_idx)
''')
    replace_once(score, '''    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):''', '''    def _get_score_ov(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """OpenVINO GenAI eviction score (SnapKV window + head-summed attention), accumulated during prefill.

        Only the last `ov_window` rows of the whole prompt (absolute positions ov_total_len - ov_window ..) score;
        their softmax attention over all keys is summed over every query head (PagedAttention `scores` output)
        into the float32 buffer self.ov_raw[layer_idx] of length ov_total_len. Chunks before the window are skipped.
        """
        from attention.ov_policy import ov_window_scores
        _, _, q_len, _ = query_states.shape
        k_total = key_states.size(2)
        seen = k_total - q_len
        lo = max(self.ov_total_len - self.ov_window, seen)
        hi = seen + q_len
        if hi <= lo:
            return
        rows = query_states[0, :, lo - seen:hi - seen]
        s = ov_window_scores(rows, key_states[0], lo)
        self.ov_raw[layer_idx][:k_total] += s

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):''')

    # --- kvcache.py: block selection in prune (EvictCache: training; RetainCache / RetainHybridCache: eval) --------
    kvc = repo / "attention/kvcache.py"
    s = kvc.read_text()
    old_evict = '''        if "uniform" in level:
            self.valid, thres = self._threshold_uniform(self.score, ratio)
        else:
            self.valid, thres = self._threshold(self.score, ratio)
        assert self.valid.size(-1) == self.ctx_len'''
    assert s.count(old_evict) == 3, "expected the prune() selection block in EvictCache, RetainCache and RetainHybridCache"
    s = s.replace(old_evict, '''        if level == "ov-block":
            self.valid, thres = self._ov_block_valid(ratio), 0.0
        elif "uniform" in level:
            self.valid, thres = self._threshold_uniform(self.score, ratio)
        else:
            self.valid, thres = self._threshold(self.score, ratio)
        assert self.valid.size(-1) == self.ctx_len''')
    helper = '''

def _ov_block_valid(self, ratio: float):
    """OpenVINO block eviction over the whole prompt (blocks aligned to position 0, the system prompt lies in the
    start area), then sliced to the context -> valid [n_layers, 1, n_heads_kv, ctx_len]."""
    from attention.ov_policy import ov_block_keep
    keep = ov_block_keep(self.ov_scores, ratio, n_restore=getattr(self, "ov_n_restore", 0), block=self.ov_block,
                         start_size=self.ov_start, recent_size=self.ov_recent)       # [L, sys + ctx]
    keep = keep[:, self.start_idx:self.end_idx]
    # same layout as _threshold(): [n_layers, batch=1, n_heads_kv, ctx_len]
    return keep[:, None, None, :].expand(-1, 1, self.n_heads_kv, -1).contiguous()


EvictCache._ov_block_valid = _ov_block_valid
RetainCache._ov_block_valid = _ov_block_valid
RetainHybridCache._ov_block_valid = _ov_block_valid
'''
    kvc.write_text(s.rstrip("\n") + "\n" + helper)

    # --- train script: CLI, level mapping, prefill scoring, budget matching ---------------------------------------
    tr = repo / "experiments/train_learnable_restore.py"
    replace_once(tr, '''    parser.add_argument("--level", choices=["pair", "pair-uniform", "snapkv", "h2o", "random"], default="pair")''',
                 '''    parser.add_argument("--level", choices=["pair", "pair-uniform", "snapkv", "h2o", "random", "ov-block"], default="pair")
    parser.add_argument("--ov-window", type=int, default=8, help="ov-block: SnapKV window (OpenVINO snapkv_window_size)")
    parser.add_argument("--ov-block", type=int, default=16, help="ov-block: KV block size (16 = OpenVINO iGPU, 32 = CPU)")
    parser.add_argument("--ov-start", type=int, default=32, help="ov-block: OpenVINO start_size")
    parser.add_argument("--ov-recent", type=int, default=32, help="ov-block: OpenVINO recent_size")''')
    replace_once(tr, '''    elif args.level == "random":
        args.score_method = "random"
        args.level = "pair"''', '''    elif args.level == "random":
        args.score_method = "random"
        args.level = "pair"
    elif args.level == "ov-block":
        args.score_method = "ov_h2o"''')
    replace_once(tr, '''    _h2o = (do_score and sm == "h2o")
    with torch.no_grad():
        if _h2o:''', '''    _h2o = (do_score and sm == "h2o")
    _ov = (do_score and sm == "ov_h2o")
    with torch.no_grad():
        if _ov:
            full_len = prefill_ids.shape[1]
            kv.get_score = True
            kv.score_mode = "ov_h2o"
            kv.ov_window, kv.ov_total_len = args.ov_window, full_len
            kv.ov_block, kv.ov_start, kv.ov_recent = args.ov_block, args.ov_start, args.ov_recent
            kv.ov_raw = [torch.zeros(full_len, dtype=torch.float32, device=kv.device) for _ in range(kv.n_layers)]
            _pf_chunk = args.prefill_chunk_size
        elif _h2o:''')
    replace_once(tr, '''        if _h2o:
            kv.get_score = False
            kv.score_mode = "kvzip"''', '''        if _ov:
            from attention.ov_policy import ov_postprocess
            kv.get_score = False
            kv.score_mode = "kvzip"
            kv.ov_scores = ov_postprocess(torch.stack(kv.ov_raw), window=args.ov_window, start_size=args.ov_start)
            del kv.ov_raw
            sys_len = model_kv.sys_prompt_ids.shape[1]
            kv.score = [s[None, None, sys_len:].expand(1, kv.n_heads_kv, -1).to(kv.dtype)
                        for s in kv.ov_scores.clamp(max=torch.finfo(kv.dtype).max)]
        elif _h2o:
            kv.get_score = False
            kv.score_mode = "kvzip"''')
    replace_once(tr, '''def budget_adjusted_prune_ratio(ratio: float, student_kv, restore_tokens, args) -> float:
    if getattr(args, "budget_mode", "budget-matched") != "budget-matched":
        return ratio''', '''def budget_adjusted_prune_ratio(ratio: float, student_kv, restore_tokens, args) -> float:
    if args.level == "ov-block":
        # OpenVINO layout: the restore slots come out of whole blocks inside ov_block_keep (same as at inference).
        matched = getattr(args, "budget_mode", "budget-matched") == "budget-matched"
        skip = os.environ.get("SKIP_RESTORE", "0") == "1"
        student_kv.ov_n_restore = restore_tokens.num_tokens if matched and not skip else 0
        return ratio
    if getattr(args, "budget_mode", "budget-matched") != "budget-matched":
        return ratio''')
    print("patched", repo)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
