#!/usr/bin/env bash
# Throughput / latency / memory inside OpenVINO GenAI: 4K input, 128-2048 output tokens, batch 1/4/8,
# full cache vs OpenVINO eviction (10%) vs + RestoreKV; then 8K input, batch 1.
set -u
cd "$(dirname "$0")/.."
: "${PYTHONPATH:?set PYTHONPATH to the patched openvino.genai build (see experiments/genai_integration/README.md)}"
mkdir -p results/genai_integration/logs
LOG=results/genai_integration/logs/genai_throughput.log
python experiments/genai_integration/genai_throughput.py --input-tokens 4096 >> $LOG 2>&1
python experiments/genai_integration/genai_throughput.py --input-tokens 8192 --outputs 128,512 --batches 1 >> $LOG 2>&1
