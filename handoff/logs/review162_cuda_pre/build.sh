#!/usr/bin/env bash
set -euo pipefail
cd /home/jetson/nano_run/r162_cuda_pre_20261120
test ! -e fused_pre.so
/usr/local/cuda/bin/nvcc -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_53,code=sm_53 \
  fused_pre.cu -o fused_pre.so > build.log 2>&1
sha256sum fused_pre.cu gpu_pre.py fused_pre.so > compiled.sha256
echo CUDA_PREPROCESSOR_BUILD_COMPLETE
