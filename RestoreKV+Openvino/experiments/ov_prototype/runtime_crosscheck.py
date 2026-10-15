"""does our Python port of OpenVINO's eviction pick the same blocks as the real OpenVINO runtime?

OpenVINO side: ContinuousBatchingPipeline (openvino.genai 2026.4.1 built from source with an env-gated log in
cache_eviction.cpp, build/genai-2026.4.1-evlog) runs each prompt with cache eviction on and max_new_tokens=2. The
first eviction happens at the first decode step (one step after prefill). The log holds, per
layer, OpenVINO's own block scores over the evictable area and the evicted logical block IDs.

Our side: OVRestoreEngine (stateful IR, same model and prompt tokens) + scoring.ov_policy_scores /
ov_select (H2O NORM_SUM) or arkv_select (ADAPTIVE_RKV), exactly as in ruler_eval.py.

Output: results/ov_prototype/<date>_runtime_crosscheck_<mode>/crosscheck.csv, one row per (sample, budget, layer): number
of evictable blocks, evicted block IDs from both sides, Jaccard overlap, identical yes/no, Spearman correlation of
the block scores and max relative score difference (H2O only; OpenVINO's ADAPTIVE_RKV log holds the raw sums).

Two phases in separate processes (two 8B models do not fit in RAM together):
    PY=build/genai-2026.4.1-evlog/build   # patched openvino_genai
    env PYTHONPATH=$PY python experiments/ov_prototype/runtime_crosscheck.py --phase ov
    env -u PYTHONPATH python experiments/ov_prototype/runtime_crosscheck.py --phase ours
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def prompts(args):
    """The exact context tokens the evaluator evicts over (chat prefix + RULER context)."""
    from transformers import AutoTokenizer
    from ruler_eval import load_samples
    tok = AutoTokenizer.from_pretrained(args.model)
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix_text = templ.split(sep)[0].split("CTX")[0]
    out = []
    for idx, row in load_samples(4096, None, args.n_per_task)[:args.n_samples]:
        ids = np.array(tok.encode(prefix_text + row["context"], add_special_tokens=False), np.int64)
        out.append((idx, row["task"], ids))
    return out


def max_cache(T: int, budget: float, block: int) -> int:
    return int(budget * T) // block * block            # same rounding as eviction.ov_select


def phase_ov(args, out_dir: Path):
    import openvino as ov
    import openvino_genai as g
    assert "evlog" in g.__file__, f"needs the patched openvino_genai build, got {g.__file__}"
    mode = {"ov": g.AggregationMode.NORM_SUM, "arkv": g.AggregationMode.ADAPTIVE_RKV}[args.mode]
    logs = out_dir / "ov_logs"
    logs.mkdir(parents=True, exist_ok=True)
    samples = prompts(args)
    for b in args.budgets:
        todo = [(i, t, ids) for i, t, ids in samples if not (logs / f"{i}_{b}.log").exists()]
        if not todo:
            continue
        # one pipeline per budget (max_cache_size is a pipeline setting); budget in tokens differs per prompt, so
        # each prompt gets its own pipeline only if lengths differ in block-rounded budget -> group by max_cache
        groups: dict[int, list] = {}
        for i, t, ids in todo:
            groups.setdefault(max_cache(len(ids), b, args.block), []).append((i, t, ids))
        for mc, items in groups.items():
            cec = g.CacheEvictionConfig(args.start, args.recent, mc, mode)
            cec.apply_rotation = False
            cec.snapkv_window_size = args.window
            sc = g.SchedulerConfig()
            sc.cache_size = 2
            sc.max_num_batched_tokens = 8192          # whole prompt in one chunk (GPU multi-chunk eviction OOMs)
            sc.dynamic_split_fuse = False
            sc.use_cache_eviction = True
            sc.cache_eviction_config = cec
            t0 = time.perf_counter()
            pipe = g.ContinuousBatchingPipeline(str(args.model), sc, args.device,
                                                {"KV_CACHE_PRECISION": "f16", "CACHE_DIR": str(ROOT / "build/ov_cache")})
            print(f"[ov] budget {b} max_cache {mc}: pipeline {time.perf_counter() - t0:.0f}s", flush=True)
            gc = g.GenerationConfig()
            gc.max_new_tokens = 2            # first eviction happens at the first decode step
            for i, t, ids in items:
                log = logs / f"{i}_{b}.log"
                os.environ["OV_GENAI_EVICTION_LOG"] = str(log) + ".part"
                t0 = time.perf_counter()
                pipe.generate([ov.Tensor(ids[None])], [gc])
                os.environ.pop("OV_GENAI_EVICTION_LOG")
                Path(str(log) + ".part").rename(log) if Path(str(log) + ".part").exists() else log.write_text("")
                print(f"[ov] idx={i} {t} T={len(ids)} budget={b}: {time.perf_counter() - t0:.1f}s, "
                      f"{sum(1 for _ in open(log))} layer lines", flush=True)
            del pipe


def parse_log(path: Path) -> dict[int, tuple[np.ndarray, set[int]]]:
    """First eviction only: layer -> (block scores over the evictable area, evicted absolute block IDs)."""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split()
        layer = int(f[1])
        if layer in out:                                  # a later eviction (none expected with max_new_tokens=1)
            break
        scores = np.array([float(x) for x in f[7].split(",")]) if len(f) > 7 and f[6] == "scores" and f[7] != "evicted" else np.zeros(0)
        ev_tok = f[f.index("evicted") + 1] if f.index("evicted") + 1 < len(f) else ""
        out[layer] = (scores, {int(x) for x in ev_tok.split(",") if x})
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _attn_sum(q: np.ndarray, k: np.ndarray, q_abs_start: int, group: int, head_dim: int) -> np.ndarray:
    """q [Hq, R, D] at positions q_abs_start.., k [Hkv, K, D] -> causal softmax attention summed over heads and rows [K]."""
    kr = np.repeat(k, group, axis=0)
    lg = np.matmul(q, kr.transpose(0, 2, 1)) / np.sqrt(head_dim)
    rows = q_abs_start + np.arange(q.shape[1])[:, None]
    lg = np.where(np.arange(k.shape[1])[None, :] > rows, -np.inf, lg)
    lg -= lg.max(-1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(-1, keepdims=True)
    return p.sum(axis=(0, 1))


def _pool(x: np.ndarray, w: int) -> np.ndarray:
    """cache_eviction.cpp _max_pool: dst[i] = max(src[i : i+w]) with the window clipped at the end."""
    if w <= 1:
        return x.copy()
    pad = np.concatenate([x, np.full(w - 1, -np.inf)])
    return np.lib.stride_tricks.sliding_window_view(pad, w).max(-1)


def runtime_replica_scores(eng, pre, T: int, mode: str, start: int, window: int, max_pool: int = 7):
    """OpenVINO's eviction state at its FIRST eviction, which happens at the first decode step (not right after
    prefill: can_generate_tokens() is false during the prefill step). Registered score rows by then:
      1. prefill: the SnapKV window (last `window` prompt rows) summed by the kernel, one registration;
      2. first decode step: the generated token's attention row over T+1 keys.
    NORM_SUM: acc = maxpool7(win) + maxpool7(dec); counters = window (W..1 for the last W) + 1; score = acc / counter.
    ADAPTIVE_RKV: queue of raw rows (max-pool 1), mean over the queue, then maxpool7 (cache_eviction.cpp
    _accumulate_with_existing_scores). Returns token scores [L, T+1] (start area +inf) and the keys [L][Hkv, T+1, D]."""
    tok = int(np.argmax(pre.logits[0]))
    dec = eng.forward(eng.embed(np.array([tok])), q_rows=1)
    keys = [k.astype(np.float32) for k, _ in eng.get_kv(dtype=np.float16, keys_only=False)]
    group = eng.n_q_heads // eng.n_kv_heads
    out = np.full((eng.n_layers, T + 1), np.inf)
    for layer in range(eng.n_layers):
        k = keys[layer]
        win = _attn_sum(pre.queries[layer][0], k[:, :T], T - window, group, eng.head_dim)[start:]      # [T - start]
        d = _attn_sum(dec.queries[layer][0], k, T, group, eng.head_dim)[start:]                         # [T + 1 - start]
        n = T - start
        if mode == "ov":
            acc = np.zeros(n + 1)
            acc[:n] = _pool(win, max_pool)
            acc += _pool(d, max_pool)
            counter = np.full(n + 1, float(window))
            counter[n - window:n] = np.arange(window, 0, -1)
            counter += 1.0
            counter[n] = 1.0
            out[layer, start:] = acc / counter
        else:
            dst = np.zeros(n + 1)
            dst[:n] = win
            dst += d
            out[layer, start:] = _pool(dst / 2.0, max_pool)
    return out, keys


def phase_ours(args, out_dir: Path):
    import openvino as ov
    from restorekv_ov import eviction as E
    from restorekv_ov.engine import OVRestoreEngine
    from restorekv_ov.safety import protect
    from restorekv_ov.scoring import ov_policy_scores
    protect(floor_gb=1.5)
    core = ov.Core()
    eng = OVRestoreEngine(args.model, args.device, checkpoint=None, core=core)
    rows = []
    for i, task, ids in prompts(args):
        T = len(ids)
        eng.reset()
        pre = eng.prefill(ids, chunk=2048, q_rows_last=args.window)
        # (a) what our evaluator does: one-shot eviction right after the context, SnapKV window only
        if args.mode == "ov":
            ev_scores = ov_policy_scores(eng, pre.queries, start_size=args.start)
        else:
            ev_scores = ov_policy_scores(eng, pre.queries, start_size=args.start, max_pool=1, norm_sum=False)
        ev_keys = [k.astype(np.float32) for k, _ in eng.get_kv(dtype=np.float16)]
        # (b) exact replica of the runtime's first eviction (one decode step later)
        rt_scores, rt_keys = runtime_replica_scores(eng, pre, T, args.mode, args.start, args.window)
        for b in args.budgets:
            log = out_dir / "ov_logs" / f"{i}_{b}.log"
            if not log.exists():
                continue
            ov_side = parse_log(log)
            mc = max_cache(T, b, args.block)
            for name, sc, keys, L_T in (("evaluator", ev_scores, ev_keys, T), ("runtime_replica", rt_scores, rt_keys, T + 1)):
                keep = (mc + 0.5) / L_T                    # reproduces max_cache exactly in the selectors
                if args.mode == "ov":
                    ev = E.ov_select(sc, keep, 0, n_kv_heads=1, block=args.block, start_size=args.start,
                                     recent_size=args.recent)[:, 0]
                else:
                    ev = E.arkv_select(sc, keys, keep, 0, n_kv_heads=1, block=args.block, start_size=args.start,
                                       recent_size=args.recent)[:, 0]
                first = args.start // args.block
                last = L_T // args.block - args.recent // args.block
                for layer in range(eng.n_layers):
                    ours_ev = {blk for blk in range(first, last) if ev[layer, blk * args.block]}
                    ov_scores, ov_ev = ov_side.get(layer, (np.zeros(0), set()))
                    ours_bs = sc[layer, first * args.block:last * args.block].reshape(-1, args.block).sum(-1)
                    n = min(len(ov_scores), len(ours_bs))
                    inter, union = len(ours_ev & ov_ev), len(ours_ev | ov_ev)
                    rel = float(np.max(np.abs(ov_scores[:n] - ours_bs[:n]) / (np.abs(ov_scores[:n]) + 1e-12))) if n else float("nan")
                    rows.append({"idx": i, "task": task, "T": T, "budget": b, "max_cache": mc, "ours": name,
                                 "layer": layer, "n_evictable_blocks_ours": last - first, "n_scores_ov_log": len(ov_scores),
                                 "n_evicted_ours": len(ours_ev), "n_evicted_ov": len(ov_ev),
                                 "jaccard": round(inter / union, 4) if union else 1.0, "identical": ours_ev == ov_ev,
                                 "spearman_block_scores": round(spearman(ov_scores[:n], ours_bs[:n]), 4) if n else "",
                                 "max_rel_score_diff": round(rel, 5),
                                 "evicted_ours": " ".join(map(str, sorted(ours_ev))),
                                 "evicted_ov": " ".join(map(str, sorted(ov_ev)))})
        print(f"[ours] idx={i} {task} T={T} done", flush=True)
    with open(out_dir / "crosscheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for name in ("runtime_replica", "evaluator"):
        for b in args.budgets:
            sel = [r for r in rows if r["budget"] == b and r["ours"] == name]
            if sel:
                summary[f"{name}@{b}"] = {
                    "layer_rows": len(sel), "identical_frac": round(float(np.mean([r["identical"] for r in sel])), 4),
                    "mean_jaccard": round(float(np.mean([r["jaccard"] for r in sel])), 4),
                    "mean_spearman": round(float(np.nanmean([r["spearman_block_scores"] if r["spearman_block_scores"] != "" else np.nan for r in sel])), 4),
                    "median_max_rel_score_diff": round(float(np.nanmedian([r["max_rel_score_diff"] for r in sel])), 5)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ov", "ours"], required=True)
    ap.add_argument("--mode", choices=["ov", "arkv"], default="ov", help="ov = H2O NORM_SUM, arkv = ADAPTIVE_RKV")
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--n-per-task", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=13)
    ap.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[0.05, 0.10, 0.25])
    ap.add_argument("--block", type=int, default=16, help="GPU block size")
    ap.add_argument("--start", type=int, default=32)
    ap.add_argument("--recent", type=int, default=32)
    ap.add_argument("--window", type=int, default=8)
    args = ap.parse_args()
    out_dir = ROOT / "results/ov_prototype" / f"{date.today()}_runtime_crosscheck_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"config_{args.phase}.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    (phase_ov if args.phase == "ov" else phase_ours)(args, out_dir)


if __name__ == "__main__":
    main()
