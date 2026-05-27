#!/usr/bin/env bash
# Reinstall PyTorch wheels matching RunPod host driver (CUDA 12.8 / cu124).
set -euo pipefail

echo "=== Before ==="
nvidia-smi | head -3 || true
python3 - <<'PY' || true
import torch
print("torch:", torch.__version__, "built_cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
PY

echo "=== Reinstalling torch for cu124 ==="
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "=== After ==="
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "built_cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA still unavailable — stop pod and redeploy with PyTorch 2.5 / CUDA 12.4 template")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "CUDA fix OK"
