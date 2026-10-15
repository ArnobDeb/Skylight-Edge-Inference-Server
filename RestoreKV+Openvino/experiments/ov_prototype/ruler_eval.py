"""RULER quality of full cache vs eviction vs eviction + RestoreKV on the OpenVINO prototype.

Methods (each at every budget = fraction of the context KEPT), each also as "<method>+restore":
  kvzip        KVzip scores, global bottom-k over (layer, kv-head, token)   <- what the released adapter was trained on
  kvzip_layer  KVzip scores, same count per layer, heads still differ (kvpress layerwise=True)
  kvzip_hs     KVzip scores, per layer, ONE token set shared by all heads (token score = max over heads)
  kvzip_block  KVzip scores in OpenVINO's cache layout: per layer, --ov-block-token blocks, heads shared, start/recent kept
  ov           OpenVINO GenAI policy: SnapKV window (--ov-window, default 8), head-summed, max-pool 7, NORM_SUM,
               --ov-block-token blocks, per layer
  arkv         OpenVINO ADAPTIVE_RKV: same window scores (raw), attention-mass 0.9 + key-diversity block selection
  ov_ph        "finer eviction" (not in OpenVINO yet): OpenVINO's scoring and block rule applied per
               (layer, KV head): scores summed per GQA group, each head keeps its own blocks (same count per head)
  ov_pht       ov_ph at token granularity (block 1): the upper reference for per-head OpenVINO-style eviction
plus "full" (no eviction). kvzip -> kvzip_block is the granularity ladder (1.7); kvzip_block vs ov isolates the scorer.
The released adapter (trained with kvzip) is mismatched for every method except "kvzip".

Eviction is applied by per-(row, layer, kv-head, token) masking inside SDPA (like kvpress), so the quality numbers
are exact for each policy. All variants of one sample decode together as batch rows (decoding is bandwidth-bound).

Prompt format and metric follow kvpress (evaluation/benchmarks/ruler). Rows are appended to results.jsonl as they
finish, so the run can be resumed; samples are taken round-robin over tasks so partial runs cover every task.

    conda activate arnob
    env -u PYTHONPATH python experiments/ov_prototype/ruler_eval.py --n-per-task 1 --out smoke2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from restorekv_ov import eviction as E  # noqa: E402
from restorekv_ov.engine import OVRestoreEngine, Variant  # noqa: E402
from restorekv_ov.safety import mem_available_gb, protect  # noqa: E402
from restorekv_ov.scoring import _AttnMaxKernel, kvzip_scores, ov_policy_scores  # noqa: E402

BASE_METHODS = ["kvzip", "kvzip_layer", "kvzip_hs", "kvzip_block", "ov"]   # default; "arkv" on request
CTRL = re.compile(r"[\x00-\x1f]")


def string_match(task: str, pred: str, refs: list[str]) -> float:
    pred = CTRL.sub("", pred.strip()).strip().lower()
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    return 100.0 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


def load_samples(seq_len: int, tasks: list[str] | None, n_per_task: int):
    """First n_per_task rows of each task, ordered round-robin over tasks."""
    import datasets
    ds = datasets.load_dataset("simonjegou/ruler", str(seq_len), split="test")
    per_task: dict[str, list] = {}
    for i, t in enumerate(ds["task"]):
        if (not tasks or t in tasks) and len(per_task.setdefault(t, [])) < n_per_task:
            per_task[t].append(i)
    order = [per_task[t][k] for k in range(n_per_task) for t in sorted(per_task) if k < len(per_task[t])]
    return [(i, ds[i]) for i in order]


def select(method: str, budget: float, n_r: int, scores: dict, eng: OVRestoreEngine, ov_start: int, ov_recent: int,
           ov_block: int = 32):
    if method == "kvzip":
        return E.kvzip_select(scores["kvzip"], budget, n_r)
    if method == "kvzip_layer":
        return E.kvzip_layer_select(scores["kvzip"], budget, n_r)
    if method == "kvzip_hs":
        return E.headshared_select(scores["kvzip"], budget, n_r)
    if method == "kvzip_block":
        return E.kvzip_block_select(scores["kvzip"], budget, n_r, block=ov_block, start_size=ov_start,
                                    recent_size=ov_recent)
    if method == "arkv":
        keys = [k.astype(np.float32) for k, _ in scores["ctx_kv"]]
        return E.arkv_select(scores["ov_raw"], keys, budget, n_r, n_kv_heads=eng.n_kv_heads, block=ov_block,
                             start_size=ov_start, recent_size=ov_recent)
    if method == "ov_ph":
        return E.ov_ph_select(scores["ov_ph"], budget, n_r, block=ov_block, start_size=ov_start, recent_size=ov_recent)
    if method == "ov_pht":
        return E.ov_ph_select(scores["ov_ph"], budget, n_r, block=1, start_size=ov_start, recent_size=ov_recent)
    if method == "ov":
        return E.ov_select(scores["ov"], budget, n_r, n_kv_heads=eng.n_kv_heads, block=ov_block, start_size=ov_start,
                           recent_size=ov_recent)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/hf/restorekv/qwen3-8b_restorekv.npz")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tasks", default="", help="comma list; empty = all 13")
    ap.add_argument("--n-per-task", type=int, default=2)
    ap.add_argument("--budgets", default="0.05,0.10,0.25")
    ap.add_argument("--methods", default=",".join(BASE_METHODS))
    ap.add_argument("--no-restore", action="store_true", help="skip the +restore variants")
    ap.add_argument("--ov-start", type=int, default=32, help="OpenVINO start_size for ov / kvzip_block")
    ap.add_argument("--ov-recent", type=int, default=32,
                    help="OpenVINO recent_size for ov / kvzip_block (default 128 in OpenVINO; 32 here so that at 4K "
                         "a 5%% budget is not consumed entirely by the start+recent areas)")
    ap.add_argument("--ov-window", type=int, default=8, help="OpenVINO snapkv_window_size for ov / arkv")
    ap.add_argument("--ov-block", type=int, default=32,
                    help="KV block size for ov / kvzip_block (OpenVINO: 32 on CPU, 16 on the iGPU). Match the adapter's "
                         "training --ov-block")
    ap.add_argument("--max-batch", type=int, default=6, help="variants decoded together (GPU memory ~0.6 GB each at 4K)")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = dataset value")
    ap.add_argument("--mem-floor-gb", type=float, default=1.5)
    ap.add_argument("--act-scale", type=float, default=8.0,
                    help="GPU ACTIVATIONS_SCALE_FACTOR (0 = off). Some adapters push the restore tokens' activations past "
                         "fp16's range on the iGPU (NaN from layer 3); training was bf16. 8 fixes it (2026-10-09)")
    ap.add_argument("--out", default="ruler4k")
    ap.add_argument("--stage", default="ov_prototype", help="results/<folder>/ the run goes to (adapter_training for adapter evaluations)")
    ap.add_argument("--date", default=str(date.today()), help="run-dir date prefix (to resume / seed a fixed dir)")
    args = ap.parse_args()
    protect(floor_gb=args.mem_floor_gb)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    encode = lambda s: np.array(tok.encode(s, add_special_tokens=False), dtype=np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix_text, question_suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]
    prefix_len = len(encode(prefix_text))
    suffix_ids = encode(question_suffix)
    stop_ids = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    methods = args.methods.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    plan = [("full", None)] + [(m + r, b) for b in budgets for m in methods for r in ([""] if args.no_restore else ["", "+restore"])]
    out_dir = ROOT / "results" / args.stage / f"{args.date}_{args.out}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "results.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["idx"], r["method"], r["budget"]))
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                      "openvino": ov.get_version(), "prompt_prefix": prefix_text,
                                                      "question_suffix": question_suffix}, indent=2))

    core = ov.Core()
    t0 = time.perf_counter()
    cc = {"ACTIVATIONS_SCALE_FACTOR": str(args.act_scale)} if args.act_scale and args.device == "GPU" else {}
    eng = OVRestoreEngine(args.model, args.device, checkpoint=args.ckpt, core=core, compile_config=cc)
    kernel = _AttnMaxKernel(core, args.device, eng.head_dim)
    print(f"engine ready in {time.perf_counter() - t0:.0f}s", flush=True)

    samples = load_samples(args.seq_len, [t for t in args.tasks.split(",") if t], args.n_per_task)
    print(f"{len(samples)} samples, {len(plan)} variants each", flush=True)
    for n_done, (idx, row) in enumerate(samples):
        todo = [(m, b) for m, b in plan if (idx, m, b) not in done]
        if not todo:
            continue
        ts = time.perf_counter()
        ctx_ids = encode(prefix_text + row["context"])
        T = len(ctx_ids)
        q_ids = encode(row["question"] + question_suffix + row["answer_prefix"])
        max_new = args.max_new_tokens or int(row["max_new_tokens"])

        eng.reset()
        t = time.perf_counter()
        pre = eng.prefill(ctx_ids, chunk=2048, q_rows_last=args.ov_window)
        timings = {"prefill_s": time.perf_counter() - t}
        scores = {}
        if any(m.startswith("kvzip") for m, _ in todo):
            t = time.perf_counter()
            scores["kvzip"] = kvzip_scores(eng, ctx_ids, prefix_len, encode, suffix_ids, kernel=kernel)
            timings["kvzip_score_s"] = time.perf_counter() - t
        if any(m.split("+")[0] == "ov" for m, _ in todo):
            t = time.perf_counter()
            scores["ov"] = ov_policy_scores(eng, pre.queries)
            timings["ov_score_s"] = time.perf_counter() - t
        if any(m.startswith("ov_ph") for m, _ in todo):
            t = time.perf_counter()
            scores["ov_ph"] = ov_policy_scores(eng, pre.queries, per_head=True)
            timings["ov_ph_score_s"] = time.perf_counter() - t
        ctx_kv = eng.get_kv(dtype=np.float16)
        if any(m.startswith("arkv") for m, _ in todo):
            t = time.perf_counter()
            scores["ov_raw"] = ov_policy_scores(eng, pre.queries, max_pool=1, norm_sum=False)   # ADAPTIVE_RKV: raw
            scores["ctx_kv"] = ctx_kv
            timings["arkv_score_s"] = time.perf_counter() - t
        restore_kv = None
        if any(m.endswith("+restore") for m, _ in todo):
            t = time.perf_counter()
            eng.restore_pass()                                   # LoRA on, over the full cache, positions T..T+n-1
            restore_kv = [(k[:, T:], v[:, T:]) for k, v in eng.get_kv(dtype=np.float16)]
            timings["restore_s"] = time.perf_counter() - t
        n_r = eng.n_restore

        variants = []
        for m, b in todo:
            restore = m.endswith("+restore")
            ev = None if m == "full" else select(m.split("+")[0], b, n_r if restore else 0, scores, eng,
                                                   args.ov_start, args.ov_recent, args.ov_block)
            variants.append(Variant(f"{m}@{b}", ev, restore))
        t = time.perf_counter()
        outputs = []
        for s in range(0, len(variants), args.max_batch):
            outputs += eng.decode_variants(ctx_kv, variants[s:s + args.max_batch], q_ids, max_new, stop_ids,
                                           restore_kv=restore_kv)
        timings["decode_all_variants_s"] = time.perf_counter() - t
        timings["sample_total_s"] = time.perf_counter() - ts

        with open(rows_path, "a") as f:
            for (m, b), v, gen in zip(todo, variants, outputs):
                pred = tok.decode([g for g in gen if g not in stop_ids], skip_special_tokens=True)
                score = string_match(row["task"], pred, list(row["answer"]))
                kept = 1.0 if v.evict_ctx is None else float(1 - v.evict_ctx.mean())
                rec = {"idx": idx, "task": row["task"], "method": m, "budget": b, "context_tokens": T,
                       "kept_context_frac": round(kept, 4), "restore_entries": n_r if v.restore else 0,
                       "score": score, "pred": pred, "answer": list(row["answer"]), "gen_tokens": len(gen),
                       "n_variants": len(variants), "mem_available_gb": round(mem_available_gb(), 1),
                       **{k: round(x, 2) for k, x in timings.items()}}
                f.write(json.dumps(rec) + "\n")
        line = " ".join(f"{m.replace('+restore', '+R')}@{b}:{string_match(row['task'], tok.decode([g for g in gen if g not in stop_ids], skip_special_tokens=True), list(row['answer'])):.0f}"
                        for (m, b), gen in zip(todo, outputs))
        print(f"[{n_done + 1}/{len(samples)} idx={idx} {row['task']} T={T} {timings['sample_total_s']:.0f}s "
              f"avail={mem_available_gb():.1f}GB] {line}", flush=True)

    summarize(rows_path, out_dir)


def summarize(rows_path: Path, out_dir: Path):
    import csv
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    keys = sorted({(r["method"], r["budget"] if r["budget"] is not None else 1.0) for r in rows},
                  key=lambda k: (k[1], k[0]))
    tasks = sorted({r["task"] for r in rows})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget_kept", "n", "avg_score"] + tasks)
        for m, b in keys:
            sel = [r for r in rows if r["method"] == m and (r["budget"] if r["budget"] is not None else 1.0) == b]
            per_task = [np.mean([r["score"] for r in sel if r["task"] == t]) if any(r["task"] == t for r in sel)
                        else "" for t in tasks]
            task_means = [x for x in per_task if x != ""]
            w.writerow([m, b, len(sel), round(float(np.mean(task_means)), 1)] +
                       [round(float(x), 1) if x != "" else "" for x in per_task])
    print((out_dir / "summary.csv").read_text())


if __name__ == "__main__":
    main()
