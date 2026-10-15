#!/usr/bin/env bash
# LOFT-32K (13 datasets x 10 queries) and RULER-32K (13 tasks x 10 samples): OpenVINO H2O eviction without and
# with RestoreKV, budgets 5/10/25%, Qwen3-8B with INT8 weights. Data: benchmarks/long_context/{build_loft.py,generate_ruler.sh}.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/eviction_study/logs
LOG=results/eviction_study/logs/long_context.log
LONG="env -u PYTHONPATH python experiments/eviction_study/long_context_eval.py --methods ov --budgets 0.05,0.10,0.25 --chunk 2048 \
      --ckpt artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz --n-per-task 10"
for spec in "loft 32k loft32k_ov_ovblock5000" "ruler 32768 ruler32k_ov_ovblock5000"; do
  set -- $spec
  for attempt in 1 2 3; do
    echo "=== $1 $2 attempt $attempt $(date)" >> "$LOG"
    $LONG --bench $1 --length $2 --out $3 >> "$LOG" 2>&1 && break
    sleep 60
  done
done
