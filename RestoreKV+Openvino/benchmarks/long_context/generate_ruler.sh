#!/usr/bin/env bash
# RULER-32K / RULER-64K (13 tasks, same set as the 4K study) generated with NVIDIA's RULER (artifacts/ref/RULER) and the
# Qwen3-8B tokenizer, as kvpress did for simonjegou/ruler (4K-16K, Llama-3.1 tokenizer). Base template (no chat
# template); our evaluator wraps context/question in Qwen3's chat template itself.
#   bash benchmarks/long_context/generate_ruler.sh [num_samples=20]
set -eo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
N=${1:-20}
export PATH=$ROOT/env/venv-data/bin:$PATH NLTK_DATA=$ROOT/env/nltk_data
unset PYTHONPATH
PY=python   # prepare.py launches "python" sub-processes, so the venv must be first in PATH
TOK=$ROOT/../models/qwen3-8b-fp16-ov
OUT=$ROOT/artifacts/data/ruler
TASKS="niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"
cd $ROOT/artifacts/ref/RULER/scripts/data
for L in 32768 65536; do
  for T in $TASKS; do
    if [ -s $OUT/$L/$T/validation.jsonl ]; then echo "skip $L $T"; continue; fi
    echo "=== $L $T $(date)"
    $PY prepare.py --save_dir $OUT/$L --benchmark synthetic --task $T --tokenizer_path $TOK --tokenizer_type hf \
        --max_seq_length $L --model_template_type base --num_samples $N
  done
done
echo "=== RULER generation finished $(date)"
