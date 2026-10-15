#!/usr/bin/env bash
# Training env for RestoreKV on Prajna (login node: has internet). Created 2026-10-08.
set -euo pipefail
A=/lustre-flash/apps/spack/opt/spack/linux-nehalem/anaconda3-2023.09-0-lys2odhvjml5katkt3d4wydfksm363ys
source $A/etc/profile.d/conda.sh
[ -d ~/restorekv/env ] || conda create -y -p ~/restorekv/env python=3.10
conda activate ~/restorekv/env
python -m pip install -U pip
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r ~/restorekv/RestoreKV/requirements.txt
pip install gdown "huggingface_hub[cli]" peft
pip install flash-attn==2.8.3 --no-build-isolation || echo "FLASH_ATTN_INSTALL_FAILED"
python - <<"PY"
import torch, transformers, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "datasets", datasets.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn missing:", e)
PY
echo SETUP_DONE
