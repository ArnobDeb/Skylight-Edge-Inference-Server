#!/usr/bin/env bash
# RULER-4K, 130 samples: KVzip's per-head eviction step by step towards OpenVINO's layout, each with and without
# the paper's released RestoreKV adapter (trained for KVzip). 32-token blocks, budgets 5/10/25%.
# Needs artifacts/hf/restorekv/qwen3-8b_restorekv.npz (see README).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/ov_prototype/logs
LOG=results/ov_prototype/logs/ruler4k_layouts.log
for attempt in 1 2 3; do      # the evaluator resumes from results.jsonl, so a retry continues where it stopped
  env -u PYTHONPATH python experiments/ov_prototype/ruler_eval.py \
      --methods kvzip,kvzip_layer,kvzip_hs,kvzip_block,ov --n-per-task 10 --max-batch 3 \
      --out ruler4k_ladder_n10 >> "$LOG" 2>&1 && break
  sleep 30
done
