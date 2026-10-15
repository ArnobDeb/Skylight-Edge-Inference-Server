#!/usr/bin/env bash
# Main comparison, RULER-4K 130 samples, 16-token blocks (iGPU), budgets 2.5/5/10/15/25%:
#   OpenVINO H2O and per-head tokens with the H2O adapter, ADAPTIVE_RKV with its adapter,
#   KVzip with the paper's adapter (the 5/10/25% KVzip numbers come from run_ruler4k_layouts.sh).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/eviction_study/logs
LOG=results/eviction_study/logs/ruler4k_main.log
EVAL="env -u PYTHONPATH python experiments/ov_prototype/ruler_eval.py --ov-block 16 --n-per-task 10 --max-batch 3 --stage eviction_study"
B5=0.025,0.05,0.10,0.15,0.25
run() {
  local out=$1; shift
  for attempt in 1 2 3; do
    echo "=== $out attempt $attempt $(date)" >> "$LOG"
    $EVAL "$@" --out "$out" >> "$LOG" 2>&1 && return
    sleep 60
  done
}
run ruler130_ovblock5000_block16 --ckpt artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz --methods ov,ov_pht --budgets $B5
run ruler130_arkv2000_block16 --ckpt artifacts/adapters/arkv_qwen3-8b/checkpoint_step2000.npz --methods arkv --budgets $B5
run ruler130_kvzip_released_extra_budgets --ckpt artifacts/hf/restorekv/qwen3-8b_restorekv.npz --methods kvzip --budgets 0.025,0.15
