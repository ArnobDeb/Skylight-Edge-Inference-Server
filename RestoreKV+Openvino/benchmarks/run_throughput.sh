#!/usr/bin/env bash
# Memory and decode speed with the KV cache physically cut to each budget (4K: fp16 model, 32K: INT8 model).
# The INT8 model is made once with benchmarks/long_context/make_int8_model.py.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/eviction_study/logs
LOG=results/eviction_study/logs/throughput.log
for c in 4k 32k; do
  env -u PYTHONPATH python experiments/eviction_study/throughput_memory.py --context $c >> "$LOG" 2>&1
done
# full cache at 32K, decoded straight from the prefill state (the copy-based path above does not fit for it)
env -u PYTHONPATH python experiments/eviction_study/throughput_memory.py --context 32k --only-full >> "$LOG" 2>&1
