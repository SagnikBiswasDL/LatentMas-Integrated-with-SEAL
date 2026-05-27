#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd /workspace
REPO_DIR="/workspace/LatentMas-Integrated-with-SEAL"
REPO_URL="https://github.com/SagnikBiswasDL/LatentMas-Integrated-with-SEAL.git"

if [[ -d "$REPO_DIR/.git" ]]; then
  cd "$REPO_DIR"
  git pull --ff-only origin main || true
else
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install vllm

chmod +x scripts/run_pilot.sh
bash scripts/run_pilot.sh
