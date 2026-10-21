#!/usr/bin/env bash
# Experiment 3: decode scaling inside OpenVINO GenAI for three models (4K input, 128-16K output, batch 1 + batch 8 at
# 2K/8K; full cache vs eviction 10% vs + RestoreKV), then the output-agreement tables. Needs the patched build on
# PYTHONPATH, the OpenVINO models in ../models and the adapters in artifacts/adapters/ovblock_<model>/.
set -u
cd "$(dirname "$0")/.."
: "${PYTHONPATH:?set PYTHONPATH to the patched openvino.genai build (see experiments/genai_integration/README.md)}"
mkdir -p results/genai_integration/logs
LOG=results/genai_integration/logs/experiment3.log
for spec in "qwen3-8b ../models/qwen3-8b-fp16-ov" "qwen3-4b ../models/qwen3-4b-fp16-ov" \
            "llama3.1-8b ../models/llama-3.1-8b-instruct-fp16-ov"; do
  set -- $spec
  M=artifacts/genai_models/$1-fp16-restorekv-ovblock5000
  [ -d $M ] || env -u PYTHONPATH python experiments/genai_integration/export_genai_model.py --model $2 \
      --ckpt artifacts/adapters/ovblock_$1/checkpoint_step5000.npz --out $M >> $LOG 2>&1
  python experiments/genai_integration/decode_scaling.py --model $M --tag $1 >> $LOG 2>&1
done
env -u PYTHONPATH python experiments/genai_integration/output_agreement.py results/genai_integration/*_decode_scaling_* >> $LOG 2>&1
