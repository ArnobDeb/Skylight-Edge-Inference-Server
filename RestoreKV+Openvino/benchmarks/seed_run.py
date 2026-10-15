"""Copy rows of finished runs into a new run directory so the evaluator (which resumes from results.jsonl) only computes
the missing variants. Rows are identical by construction: same samples, same deterministic greedy decoding, same
eviction settings for the copied methods.

    python benchmarks/seed_run.py <new_run_dir> <src_run_dir>:<method>[,<method>...] [...]
e.g. seed the plain `ov` and `full` rows of the 2026-10-09 sweep into a new run that adds budgets / methods.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    dst = Path(sys.argv[1])
    dst.mkdir(parents=True, exist_ok=True)
    rows_path = dst / "results.jsonl"
    have = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            have.add((r["idx"], r["method"], r["budget"]))
    n = 0
    with open(rows_path, "a") as f:
        for spec in sys.argv[2:]:
            src, methods = spec.rsplit(":", 1)
            keep = set(methods.split(","))
            for line in (Path(src) / "results.jsonl").read_text().splitlines():
                r = json.loads(line)
                key = (r["idx"], r["method"], r["budget"])
                if r["method"] in keep and key not in have:
                    r["seeded_from"] = Path(src).name
                    f.write(json.dumps(r) + "\n")
                    have.add(key)
                    n += 1
    print(f"seeded {n} rows into {rows_path}")


if __name__ == "__main__":
    main()
