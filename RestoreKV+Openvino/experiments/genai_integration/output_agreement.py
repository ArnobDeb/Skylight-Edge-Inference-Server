"""Experiment 3B: how close do evicted generations stay to the full-cache generation?

Reads the generations.jsonl written by decode_scaling.py (same prompt, greedy decoding, ignore_eos) and compares,
for every output length and batch-1 point, the `ov` and `ov+R` token sequences with the `full` one:
    first_divergence     index of the first token that differs (= length of the identical prefix)
    positional_match     share of positions with the same token
    unigram_f1 / 4gram_f1  bag-of-n-gram overlap (F1) between the two outputs, a cheap ROUGE-N
No GPU needed.

    python experiments/genai_integration/output_agreement.py results/genai_integration/2026-10-20_decode_scaling_qwen3-8b

Output: agreement.csv in the same run folder.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def ngram_f1(a: list[int], b: list[int], n: int) -> float:
    ca = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1))
    cb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p, r = overlap / max(1, sum(ca.values())), overlap / max(1, sum(cb.values()))
    return 2 * p * r / (p + r)


def compare(ref: list[int], hyp: list[int]) -> dict:
    n = min(len(ref), len(hyp))
    first = next((i for i in range(n) if ref[i] != hyp[i]), n)
    return {"first_divergence": first, "positional_match": round(sum(ref[i] == hyp[i] for i in range(n)) / max(1, n), 4),
            "unigram_f1": round(ngram_f1(ref, hyp, 1), 4), "4gram_f1": round(ngram_f1(ref, hyp, 4), 4)}


def main(run_dir: Path):
    gens = {}
    for line in open(run_dir / "generations.jsonl"):
        g = json.loads(line)
        if g["batch"] == 1:
            gens[(g["config"], g["output_tokens"])] = g["ids"]
    rows = []
    for (cfg, out), ids in sorted(gens.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if cfg == "full" or ("full", out) not in gens:
            continue
        rows.append({"config": cfg, "output_tokens": out, **compare(gens[("full", out)], ids)})
    with open(run_dir / "agreement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "output_tokens", "first_divergence", "positional_match",
                                          "unigram_f1", "4gram_f1"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(Path(d))
