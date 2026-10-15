#!/bin/bash
#SBATCH -J rkv_train
#SBATCH -p a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/rethinkingai-self/25m0779/restorekv/logs/train_%x_%j.out
# RestoreKV adapter training on Prajna (official code, branch ov-policy). Paper recipe: 8 restore tokens,
# LoRA r8/a16, lr 2e-4, symmetric KL self-distillation, budget ratio ~ U(0.025, 0.25), budget-matched.
#   LEVEL=pair      -> paper's KVzip evictor (sanity check: should reproduce the released adapter)
#   LEVEL=ov-block  -> OpenVINO GenAI evictor (SnapKV-window H2O / NORM_SUM, 16-token blocks, heads shared)
#   LEVEL=ov-arkv   -> OpenVINO ADAPTIVE_RKV (attention mass 0.9 + key diversity, 16-token blocks, heads shared)
#   LEVEL=ov-headblock -> per-head blocks ("finer eviction"): OpenVINO scoring/blocks per (layer, KV head)
#   OV_WINDOW=8     -> OpenVINO snapkv_window_size (SnapKV variants: 16, 32, ...)
#   OV_RECENT=32    -> OpenVINO recent_size (OpenVINO default 128)
#   OV_BLOCK=16     -> block size; 1 with ov-headblock = per-head TOKENS (finest eviction)
#   N_RESTORE=8     -> number of restore tokens (1..16 cost the same one 16-token block)
# Submit (from ~/restorekv/jobs):
#   sbatch --export=ALL,LEVEL=ov-block,STEPS=5000,NAME=ovblock_qwen3-8b -J rkv_ovblock train.sbatch
#   sbatch --export=ALL,LEVEL=pair,STEPS=30,NAME=smoke_pair -J rkv_smoke -t 01:00:00 train.sbatch
set -eo pipefail   # no -u: conda activate.d scripts read unset variables
: "${LEVEL:=ov-block}" "${STEPS:=5000}" "${MODEL:=qwen3-8b}" "${NAME:=${LEVEL}_${MODEL}}" "${EXTRA:=}"
A=/lustre-flash/apps/spack/opt/spack/linux-nehalem/anaconda3-2023.09-0-lys2odhvjml5katkt3d4wydfksm363ys
source "$A/etc/profile.d/conda.sh"; conda activate ~/restorekv/env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=~/restorekv/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/restorekv/RestoreKV
echo "branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) | LEVEL=$LEVEL OV_WINDOW=${OV_WINDOW:-8} OV_RECENT=${OV_RECENT:-32} OV_BLOCK=${OV_BLOCK:-16} N_RESTORE=${N_RESTORE:-8} STEPS=$STEPS NAME=$NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OV_ARGS=""
case "$LEVEL" in ov-block|ov-arkv|ov-headblock) OV_ARGS="--ov-window ${OV_WINDOW:-8} --ov-block ${OV_BLOCK:-16} --ov-start 32 --ov-recent ${OV_RECENT:-32}";; esac
python experiments/train_learnable_restore.py \
  --model "$MODEL" --data-source teacher-responses \
  --teacher-responses-path "data/sft_data/${MODEL}_train_mix.jsonl" \
  --output-dir "$HOME/restorekv/runs/$NAME" \
  --level "$LEVEL" $OV_ARGS --num-restore-tokens ${N_RESTORE:-8} --lora-rank 8 --lora-alpha 16 \
  --restore-lr 2e-4 --lora-lr 2e-4 --max-steps "$STEPS" --val-every 500 --save-every 500 \
  --kl-mode symmetric --distill-alpha 1.0 --ratio-min 0.025 --ratio-max 0.25 \
  --budget-mode budget-matched --seed 0 $EXTRA
echo TRAIN_DONE
