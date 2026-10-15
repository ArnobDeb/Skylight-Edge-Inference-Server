"""LOFT-32K / LOFT-64K for the RestoreKV evaluator (Qwen3 tokenizer), built with LOFT's own prompt code
(artifacts/ref/LOFT, Corpus-in-Context prompts, few_shot_with_cot) from the downloaded data (artifacts/data/loft).

Datasets (text, no infilling needed): RAG nq / hotpotqa / musique (subspan EM), retrieval nq (recall@1),
hotpotqa (mrecall@2), scifact (recall@1). The 10 dev queries of a dataset share one context (instructions + corpus +
few-shot examples), so the evaluator prefills, scores and restores once per dataset and answers all 10 queries.
ICL is left out: its "32k" sets are ~9K Qwen tokens.

64K: LOFT ships 32k / 128k / 1m only. The 64K corpus = every document of the 32k corpus (so all gold documents and
few-shot documents are present) plus further documents of the 128k corpus that are not in the 32k one (in 128k file order), spread evenly between the 32k
documents (gold documents stay spread through the context), until the whole prompt reaches --target-64k tokens. Same queries and
few-shot examples as 32k. Built as a separate LOFT base dir whose folders are named "32k" so LOFT's prompt registry
can be reused unchanged.

    env -u PYTHONPATH env/venv-data/bin/python benchmarks/long_context/build_loft.py
Output: artifacts/data/loft_qwen/{32k,64k}.jsonl (one row per query: context, question, answers, metric, ...)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOFT = ROOT / "artifacts/ref/LOFT"
DATA = ROOT / "artifacts/data/loft"
OUT = ROOT / "artifacts/data/loft_qwen"
DATASETS = [  # (task, dataset, metric, top_k, max_new_tokens)
    ("rag", "nq", "subspan_em", 0, 160),
    ("rag", "hotpotqa", "subspan_em", 0, 192),
    ("rag", "musique", "subspan_em", 0, 224),
    ("retrieval", "nq", "recall@k", 1, 96),
    ("retrieval", "hotpotqa", "mrecall@k", 2, 128),
    ("retrieval", "scifact", "recall@k", 1, 96),
    # added 2026-10-14 so LOFT-32K has 13 datasets x 10 queries = 130 queries, like RULER-4K's 130 samples
    ("retrieval", "arguana", "recall@k", 1, 96),
    ("retrieval", "fever", "recall@k", 1, 96),
    ("retrieval", "musique", "mrecall@k", 5, 160),
    ("retrieval", "qampari", "mrecall@k", 5, 160),
    ("retrieval", "quest", "mrecall@k", 3, 160),
    ("rag", "qampari", "multi_subspan_em", 0, 256),
    ("rag", "quest", "multi_subspan_em", 0, 256),
]


def build_64k_dir(tok, task: str, ds: str, target: int, base64: Path, ctx32_tokens: int) -> None:
    src32, src128 = DATA / "data" / task / ds / "32k", DATA / "data" / task / ds / "128k"
    dst = base64 / "data" / task / ds / "32k"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("dev_queries.jsonl", "few_shot_queries.jsonl"):
        shutil.copy(src32 / f, dst / f)
    docs32 = [json.loads(l) for l in open(src32 / "corpus.jsonl")]
    pids32 = {d["pid"] for d in docs32}
    docs128 = [json.loads(l) for l in open(src128 / "corpus.jsonl")]
    need = target - ctx32_tokens
    extras, extra_tok = [], 0
    for d in docs128:
        if extra_tok >= need:
            break
        if d["pid"] not in pids32:
            extras.append(d)
            extra_tok += len(tok.encode(f"ID: 0000 | TITLE: {d.get('title_text', '')} | CONTENT: {d.get('passage_text', '')}\n"))
    # interleave: the extra documents are spread evenly between the 32k documents (gold documents stay spread out)
    merged, j = [], 0
    for i, d in enumerate(docs32):
        merged.append(d)
        upto = round((i + 1) * len(extras) / len(docs32))
        merged += extras[j:upto]
        j = upto
    with open(dst / "corpus.jsonl", "w") as f:
        for d in merged:
            f.write(json.dumps(d) + "\n")


def examples(task: str, ds: str, base_dir: Path):
    sys.path.insert(0, str(LOFT))
    import prompts  # noqa: F401  (registers all prompts)
    import utils
    from prompts import prompt_registry
    name = f"{task}_{ds}_32k_dev:few_shot_with_cot"
    exs = prompt_registry.PromptRegistry.get_examples(name=name, base_dir=str(base_dir))
    rows = []
    for e in exs:
        rows.append({"qid": e.qid, "context": utils.concatenate_chunks(e.context_chunks),
                     "question": utils.concatenate_chunks(e.query_turns[0].chunks),
                     "answers": [list(map(str, a)) for a in e.gold_answers][0]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=ROOT.parent / "models/qwen3-8b-fp16-ov")
    ap.add_argument("--target-64k", type=int, default=60500, help="prompt tokens for the 64K version")
    ap.add_argument("--lengths", default="32k,64k")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    OUT.mkdir(parents=True, exist_ok=True)
    base64 = ROOT / "artifacts/data/loft64"
    for length in args.lengths.split(","):
        out_rows = []
        for task, ds, metric, k, max_new in DATASETS:
            if length == "64k":
                ctx32 = len(tok.encode(examples(task, ds, DATA)[0]["context"]))
                build_64k_dir(tok, task, ds, args.target_64k, base64, ctx32)
            rows = examples(task, ds, DATA if length == "32k" else base64)
            n_tok = len(tok.encode(rows[0]["context"]))
            print(f"{length} {task}_{ds}: {len(rows)} queries, context {n_tok} tokens", flush=True)
            for r in rows:
                out_rows.append({**r, "task": f"{task}_{ds}", "task_type": task, "metric": metric, "top_k": k,
                                 "max_new_tokens": max_new, "answer_prefix": "", "context_tokens": n_tok})
        with open(OUT / f"{length}.jsonl", "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {OUT / (length + '.jsonl')}: {len(out_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
