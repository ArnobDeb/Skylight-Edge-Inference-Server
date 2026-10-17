#!/usr/bin/env bash
# RULER-4K, 130 samples, inside OpenVINO GenAI (patched build on PYTHONPATH): OpenVINO eviction vs + RestoreKV at
# 2.5/5/10/15/25%, then the full-cache reference. Needs the model folder from export_genai_model.py.
set -u
cd "$(dirname "$0")/.."
: "${PYTHONPATH:?set PYTHONPATH to the patched openvino.genai build (see experiments/genai_integration/README.md)}"
mkdir -p results/genai_integration/logs
LOG=results/genai_integration/logs/genai_ruler4k.log
M=artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000
[ -d $M ] || python experiments/genai_integration/export_genai_model.py --model ../models/qwen3-8b-fp16-ov \
    --ckpt artifacts/adapters/ovblock_qwen3-8b/checkpoint_step5000.npz --out $M
python experiments/genai_integration/genai_ruler_eval.py --model $M --n-per-task 10 --out genai_ruler4k_n10 >> $LOG 2>&1
python experiments/genai_integration/genai_ruler_eval.py --model $M --n-per-task 10 --no-eviction --out genai_ruler4k_n10_full >> $LOG 2>&1
