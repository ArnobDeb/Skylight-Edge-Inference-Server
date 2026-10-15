#!/bin/bash
#SBATCH -J rkv_build
#SBATCH -p a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /home/rethinkingai-self/25m0779/restorekv/logs/build_ext_%j.out
# Build flash-attn 2.8.3 (prebuilt wheels need glibc 2.32; Rocky 8 has 2.28) and the RestoreKV csrc kernel,
# offline on a compute node, then smoke-test both on the GPU.
set -eo pipefail  # no -u: conda gcc activate.d scripts read unset vars
A=/lustre-flash/apps/spack/opt/spack/linux-nehalem/anaconda3-2023.09-0-lys2odhvjml5katkt3d4wydfksm363ys
source $A/etc/profile.d/conda.sh; conda activate ~/restorekv/env
# conda-forge gcc 12 (spack gcc 14.2 is built for icelake and dies with SIGILL on the a40 nodes)
export CUDA_HOME=$(ls -d ~/spack/linux-nehalem/cuda-12.9.0-*)
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export PATH=$CUDA_HOME/bin:$PATH NVCC_PREPEND_FLAGS="-ccbin $CXX"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export MAX_JOBS=12 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=80 PIP_NO_INDEX=1
nvidia-smi --query-gpu=name --format=csv,noheader; nvcc --version | tail -1; $CXX --version | head -1
date; pip install --no-build-isolation --no-deps ~/restorekv/src_pkgs/flash_attn-2.8.3.tar.gz; date
cd ~/restorekv/RestoreKV/csrc && python build.py install; date
cd ~ && python - <<"PY"
import torch, flash_attn, tiny_api_cuda
from flash_attn import flash_attn_varlen_func
q = torch.randn(64, 8, 128, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
o = flash_attn_varlen_func(q, q, q, cu, cu, 64, 64, causal=True)
print("flash_attn", flash_attn.__version__, "varlen ok", tuple(o.shape), "| tiny_api_cuda ok")
PY
echo BUILD_EXT_DONE
