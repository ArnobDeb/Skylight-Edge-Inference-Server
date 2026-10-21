#!/usr/bin/env bash
# RULER-4K 130 samples inside OpenVINO GenAI under the question-blind protocol (RestoreKV paper / prototype):
# inside order, then exact order. Needs the patched openvino.genai build on PYTHONPATH.
set -u
cd "$(dirname "$0")/.."
: "${PYTHONPATH:?set PYTHONPATH to the patched openvino.genai build (see experiments/genai_integration/README.md)}"
mkdir -p results/genai_integration/logs
LOG=results/genai_integration/logs/genai_question_blind.log
M=artifacts/genai_models/qwen3-8b-fp16-restorekv-ovblock5000
python experiments/genai_integration/genai_ruler_eval.py --model $M --n-per-task 10 --query-blind --out genai_ruler4k_qblind_n10 >> $LOG 2>&1
python experiments/genai_integration/genai_ruler_eval.py --model $M --n-per-task 10 --query-blind --restore-order exact --out genai_ruler4k_qblind_exact_n10 >> $LOG 2>&1
