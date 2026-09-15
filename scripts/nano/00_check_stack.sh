#!/usr/bin/env bash
# Kiem tra stack JetPack / CUDA / cuDNN / TensorRT / Python tren Jetson Nano. Chay TREN THIET BI.
set -u
OUT=${1:-results/phase2/profile/nano_stack.txt}
mkdir -p "$(dirname "$OUT")"
{
  echo "== date: $(date -Is)"; echo "== hostname: $(hostname)"
  echo "== L4T:"; cat /etc/nv_tegra_release 2>/dev/null || echo "(khong co /etc/nv_tegra_release)"
  echo "== jetson_release:"; jetson_release 2>/dev/null || echo "(cai jetson-stats: sudo -H pip3 install jetson-stats)"
  echo "== python3: $(python3 --version 2>&1)"
  echo "== CUDA:"; ls /usr/local/ | grep -i cuda; nvcc --version 2>/dev/null | tail -2
  echo "== dpkg tensorrt/cudnn/cuda:"; dpkg -l | grep -E "tensorrt|libnvinfer|cudnn|cuda-toolkit" | awk '{print $2, $3}'
  echo "== trtexec:"; ls /usr/src/tensorrt/bin/trtexec 2>/dev/null && /usr/src/tensorrt/bin/trtexec --help 2>/dev/null | head -1
  echo "== python packages:"; python3 -c "import tensorrt; print('tensorrt', tensorrt.__version__)" 2>&1
  python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1
  python3 -c "import cv2; print('cv2', cv2.__version__)" 2>&1
  python3 -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())" 2>&1
  python3 -c "import pycuda; print('pycuda ok')" 2>&1
  echo "== nvpmodel:"; sudo nvpmodel -q 2>/dev/null || echo "(can sudo)"
  echo "== memory:"; free -m
  echo "== tegrastats (1 mau):"; timeout 2 tegrastats 2>/dev/null | head -1
} | tee "$OUT"
