#!/usr/bin/env bash
# Login node only (compute nodes have no internet). Qwen3-8B weights -> HF cache; RestoreKV training data -> repo.
set -uo pipefail
A=/lustre-flash/apps/spack/opt/spack/linux-nehalem/anaconda3-2023.09-0-lys2odhvjml5katkt3d4wydfksm363ys
source $A/etc/profile.d/conda.sh; conda activate ~/restorekv/env
export HF_HOME=~/restorekv/hf_cache
huggingface-cli download Qwen/Qwen3-8B --exclude "*.pth" && echo QWEN3_8B_DONE
mkdir -p ~/restorekv/data && cd ~/restorekv/data
gdown 1UUfIPS16YAqegaFAInRL6qeTuGxBWfDm -O restorekv_train_data.zip && echo DATA_ZIP_DONE
unzip -o -q restorekv_train_data.zip -d ~/restorekv/RestoreKV/ && ls -la ~/restorekv/RestoreKV/data/sft_data && echo DATA_DONE
