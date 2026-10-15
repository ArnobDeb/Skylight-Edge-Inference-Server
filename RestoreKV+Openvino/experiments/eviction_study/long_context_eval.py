"""RULER / LOFT at 32K and 64K on the OpenVINO prototype: full cache vs eviction vs eviction + RestoreKV.

Same methods and selection code as the 4K evaluator (experiments/ov_prototype/ruler_eval.py), but built for
long contexts on a 30 GB laptop:
  * no host copies of the KV cache: one prefill per context, then every variant decodes on the SAME device state.
    Eviction is by masking (as in the 4K study); the tokens of earlier variants stay in the state and are masked,
    restore entries are masked for variants without RestoreKV, question positions restart at T (+n restore);
  * keys are read one layer at a time for scoring;
  * LOFT queries that share a context (all 10 dev queries of a dataset) share the prefill, scoring and restore pass;
  * model: INT8 weights (artifacts/models/qwen3-8b-int8-ov) and, for 64K, YaRN RoPE scaling (--rope-factor 2).
Methods: full, ov (OpenVINO H2O NORM_SUM, head-shared blocks), ov_ph (per-head blocks), ov_pht (per-head tokens),
each also +restore. ADAPTIVE_RKV / KVzip are left out at long context (O(T^2) diversity matrix / context replay).

    env -u PYTHONPATH python experiments/eviction_study/long_context_eval.py --bench ruler --length 32768 \
        --ckpt artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz --n-per-task 1 --out smoke
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import string
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/ov_prototype"))
from restorekv_ov import eviction as E  # noqa: E402
from restorekv_ov.engine import OVRestoreEngine  # noqa: E402
from restorekv_ov.safety import mem_available_gb, protect  # noqa: E402
from restorekv_ov.scoring import ov_policy_scores  # noqa: E402
from ruler_eval import string_match, summarize  # noqa: E402

# RULER question / answer split, as kvpress evaluation/benchmarks/ruler/create_huggingface_dataset.py
QUESTION_PATTERNS = {
    "niah": re.compile(r"What (?:is|are all) the special magic"),
    "vt": re.compile(r"Question: Find all variables that are assigned the value"),
    "cwe": re.compile(r"Question: What are the 10 most common words in the above list\?"),
    "fwe": re.compile(r"Question: Do not provide any explanation\."),
    "qa": re.compile(r"Answer the question based on the given documents\."),
}
MAX_NEW = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
               "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]


def load_ruler(length: int, n_per_task: int, tasks: list[str] | None):
    per_task = {}
    for task in tasks or RULER_TASKS:
        rows = [json.loads(l) for l in open(ROOT / f"artifacts/data/ruler/{length}/{task}/validation.jsonl")]
        out = []
        for r in rows[:n_per_task]:
            fam = task.split("_")[0]
            text = r["input"]
            i = list(QUESTION_PATTERNS[fam].finditer(text))[-1].start()
            out.append({"idx": f"{task}:{r['index']}", "task": task, "context": text[:i], "question": text[i:],
                        "answer_prefix": r.get("answer_prefix", ""), "answers": list(r["outputs"]),
                        "metric": "ruler", "max_new_tokens": MAX_NEW[fam]})
        per_task[task] = out
    return [per_task[t][k] for k in range(n_per_task) for t in per_task if k < len(per_task[t])]


def load_loft(length: str, n_per_task: int, tasks: list[str] | None):
    rows = [json.loads(l) for l in open(ROOT / f"artifacts/data/loft_qwen/{length}.jsonl")]
    by_task = {}
    for r in rows:
        if tasks and r["task"] not in tasks:
            continue
        if len(by_task.setdefault(r["task"], [])) < n_per_task:
            by_task[r["task"]].append({**r, "idx": f"{r['task']}:{r['qid']}"})
    return [r for t in by_task for r in by_task[t]]          # grouped by task: consecutive queries share the context


# --- LOFT metrics (artifacts/ref/LOFT/evaluation/{utils,rag,retrieval}.py) ---------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract(pred: str) -> list:
    for line in pred.replace("*", "").replace("`", "").strip().split("\n"):
        if "[" in line and "]" in line:
            try:
                v = ast.literal_eval(re.sub(r"([a-zA-Z0-9])'([a-zA-Z0-9])", r"\1\\'\2", line[line.find("["):line.rfind("]") + 1]))
                return list(v) if isinstance(v, (list, tuple)) else [v]
            except Exception:  # noqa: BLE001
                return []
    return []


def score(row: dict, pred: str) -> float:
    if row["metric"] == "ruler":
        return string_match(row["task"], pred, row["answers"])
    preds = _extract(pred)
    if not preds:
        return 0.0
    if row["metric"] == "subspan_em":
        p = _norm(str(preds[0]))
        return 100.0 * max(float(_norm(g) in p) for g in row["answers"])
    if row["metric"] == "multi_subspan_em":            # LOFT multi_value_rag: every gold answer matched by a distinct prediction
        from scipy.optimize import linear_sum_assignment
        gold = [_norm(str(g)) for g in row["answers"]]
        pr = [_norm(str(x)) for x in preds]
        m = np.array([[float(g in q or q in g) for q in pr] for g in gold])
        r, c = linear_sum_assignment(-m)
        aligned = np.zeros(len(gold))
        aligned[r] = m[r, c]
        return 100.0 * float(aligned.all())
    gold = {str(g).strip() for g in row["answers"]}
    top = {str(x).strip() for x in preds[:row["top_k"]]}
    hit = len(top & gold)
    if row["metric"] == "recall@k":
        return 100.0 * hit / len(gold)
    return 100.0 * float(hit == min(row["top_k"], len(gold)))              # mrecall@k


def select(method: str, budget: float, n_r: int, scores: dict, args):
    if method == "ov":
        return E.ov_select(scores["ov"], budget, n_r, n_kv_heads=8, block=args.ov_block, start_size=args.ov_start,
                           recent_size=args.ov_recent)
    if method == "ov_ph":
        return E.ov_ph_select(scores["ov_ph"], budget, n_r, block=args.ov_block, start_size=args.ov_start,
                              recent_size=args.ov_recent)
    if method == "ov_pht":
        return E.ov_ph_select(scores["ov_ph"], budget, n_r, block=1, start_size=args.ov_start,
                              recent_size=args.ov_recent)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["ruler", "loft"], required=True)
    ap.add_argument("--length", required=True, help="ruler: 32768 / 65536; loft: 32k / 64k")
    ap.add_argument("--model", type=Path, default=ROOT / "artifacts/models/qwen3-8b-int8-ov")
    ap.add_argument("--rope-factor", type=float, default=None, help="YaRN factor (2 for 64K); default: auto")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--n-per-task", type=int, default=3)
    ap.add_argument("--budgets", default="0.05,0.10,0.25")
    ap.add_argument("--methods", default="ov,ov_ph")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--ov-block", type=int, default=16)
    ap.add_argument("--ov-start", type=int, default=32)
    ap.add_argument("--ov-recent", type=int, default=128, help="OpenVINO default 128 (fine at 32K/64K)")
    ap.add_argument("--ov-window", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1024, help="prefill chunk")
    ap.add_argument("--act-scale", type=float, default=8.0)
    ap.add_argument("--mem-floor-gb", type=float, default=1.5)
    ap.add_argument("--stage", default="eviction_study")
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    protect(floor_gb=args.mem_floor_gb)
    if args.rope_factor is None:
        args.rope_factor = 2.0 if args.length in ("65536", "64k") else None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    encode = lambda s: np.array(tok.encode(s, add_special_tokens=False), dtype=np.int64)  # noqa: E731
    sep = "#" * 64
    templ = tok.apply_chat_template([{"role": "user", "content": "CTX" + sep}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
    prefix_text, question_suffix = templ.split(sep)[0].split("CTX")[0], templ.split(sep)[1]
    stop_ids = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    tasks = [t for t in args.tasks.split(",") if t]
    samples = (load_ruler(int(args.length), args.n_per_task, tasks) if args.bench == "ruler"
               else load_loft(args.length, args.n_per_task, tasks))
    methods = args.methods.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]
    plan = [("full", None)] + [(m + r, b) for b in budgets for m in methods
                               for r in ([""] if args.no_restore else ["", "+restore"])]
    out_dir = ROOT / "results" / args.stage / f"{args.date}_{args.out}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "results.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["idx"], r["method"], r["budget"]))
    (out_dir / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                      "openvino": ov.get_version()}, indent=2))

    t0 = time.perf_counter()
    cc = {"ACTIVATIONS_SCALE_FACTOR": str(args.act_scale)} if args.act_scale and args.device == "GPU" else {}
    eng = OVRestoreEngine(args.model, args.device, checkpoint=args.ckpt, core=ov.Core(), compile_config=cc,
                          rope_factor=args.rope_factor)
    n_r = eng.n_restore
    print(f"engine ready in {time.perf_counter() - t0:.0f}s (rope factor {args.rope_factor}, "
          f"attention factor {eng.rope_attention_factor:.4f}); {len(samples)} samples x {len(plan)} variants",
          flush=True)

    ctx_key, T, scores, timings_ctx = None, 0, {}, {}
    snap, snap_len = None, 0
    for n_done, row in enumerate(samples):
        todo = [(m, b) for m, b in plan if (row["idx"], m, b) not in done]
        if not todo:
            continue
        ts = time.perf_counter()
        if row["context"] != ctx_key:                       # new context: prefill, score, restore once
            ctx_ids = encode(prefix_text + row["context"])
            T = len(ctx_ids)
            eng.fresh_request()                  # free the previous document's GPU buffers (see engine)
            t = time.perf_counter()
            pre = eng.prefill(ctx_ids, chunk=args.chunk, q_rows_last=args.ov_window)
            timings_ctx = {"prefill_s": time.perf_counter() - t}
            scores = {}
            t = time.perf_counter()
            if any(m.split("+")[0] == "ov" for m, _ in plan):
                scores["ov"] = ov_policy_scores(eng, pre.queries, start_size=args.ov_start)
            if any(m.startswith("ov_ph") for m, _ in plan):
                scores["ov_ph"] = ov_policy_scores(eng, pre.queries, start_size=args.ov_start, per_head=True)
            timings_ctx["score_s"] = time.perf_counter() - t
            if not args.no_restore:
                t = time.perf_counter()
                eng.restore_pass()                          # LoRA on, over the full cache, positions T..T+n-1
                timings_ctx["restore_s"] = time.perf_counter() - t
            ctx_key = row["context"]
            # snapshot of document + restore entries; every later question of this document starts from it, so the
            # answers of earlier questions do not pile up in the cache (LOFT: 10 questions per document)
            snap = eng.get_kv(dtype=np.float16)
            snap_len = eng.kv_len
        elif eng.kv_len > snap_len:
            eng.set_kv(snap, positions_next=snap_len)       # no reset() before set_kv: GPU plugin quirk
        q_ids = encode(row["question"] + question_suffix + row["answer_prefix"])
        recs = []
        for m, b in todo:
            restore = m.endswith("+restore")
            base_len = eng.kv_len                           # T + n_r + tokens of earlier variants
            evict = np.zeros((eng.n_layers, eng.n_kv_heads, base_len), bool)
            if m != "full":
                evict[:, :, :T] = select(m.split("+")[0], b, n_r if restore else 0, scores, args)
            n_res = 0 if args.no_restore else n_r
            if not restore:
                evict[:, :, T:T + n_res] = True
            evict[:, :, T + n_res:] = True                  # earlier variants' question/answer tokens
            eng.mask_entries(evict)
            eng.next_pos = np.array([T + (n_r if restore else 0)], np.int64)
            t = time.perf_counter()
            logits = eng.forward(eng.embed(q_ids)).logits
            ttft_q = time.perf_counter() - t
            gen = eng._greedy_batch(logits, row["max_new_tokens"], stop_ids)[0]
            dt = time.perf_counter() - t
            pred = tok.decode([g for g in gen if g not in stop_ids], skip_special_tokens=True)
            kept = 1.0 if m == "full" else float(1 - evict[:, :, :T].mean())
            recs.append({"idx": row["idx"], "task": row["task"], "method": m, "budget": b, "context_tokens": T,
                         "kept_context_frac": round(kept, 4), "restore_entries": n_r if restore else 0,
                         "score": score(row, pred), "pred": pred, "answer": row["answers"], "gen_tokens": len(gen),
                         "question_prefill_s": round(ttft_q, 2), "decode_s": round(dt, 2),
                         "tpot_s": round((dt - ttft_q) / max(1, len(gen) - 1), 3),
                         "state_len": int(eng.kv_len), "mem_available_gb": round(mem_available_gb(), 1),
                         **{k: round(x, 2) for k, x in timings_ctx.items()}})
        with open(rows_path, "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        line = " ".join(f"{r['method'].replace('+restore', '+R')}@{r['budget']}:{r['score']:.0f}" for r in recs)
        print(f"[{n_done + 1}/{len(samples)} {row['idx']} T={T} {time.perf_counter() - ts:.0f}s "
              f"avail={mem_available_gb():.1f}GB] {line}", flush=True)

    summarize(rows_path, out_dir)


if __name__ == "__main__":
    main()
