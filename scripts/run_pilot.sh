#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# RunPod PyTorch 2.11+cu13 images often mismatch host driver 12.8 — fix if needed.
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "CUDA unavailable — running fix_runpod_cuda.sh"
  bash "$ROOT/scripts/fix_runpod_cuda.sh"
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
TASK="${TASK:-gsm8k}"
SAMPLES="${SAMPLES:-100}"
LATENT_STEPS="${LATENT_STEPS:-40}"
SEAL_LAYER="${SEAL_LAYER:-28}"
SEAL_COEF="${SEAL_COEF:-1.0}"
VECTOR_PATH="${VECTOR_PATH:-artifacts/seal_vectors/qwen3-14b/layer_${SEAL_LAYER}_steervec.pt}"

mkdir -p results/pilot

if [[ -z "${TMUX:-}" && "${ALLOW_NO_TMUX:-0}" != "1" ]]; then
  echo "ERROR: Not inside tmux — SSH/WiFi drops will kill this job."
  echo ""
  echo "  tmux new -s pilot"
  echo "  cd $ROOT && bash scripts/run_pilot.sh"
  echo "  # Detach: Ctrl+B, then D"
  echo ""
  echo "Reattach later: tmux attach -t pilot"
  echo "Override (not recommended): ALLOW_NO_TMUX=1 bash scripts/run_pilot.sh"
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
echo "Detected ${GPU_COUNT} GPU(s)"

COMMON_ARGS=(
  --model_name "$MODEL"
  --task "$TASK"
  --prompt sequential
  --latent_steps "$LATENT_STEPS"
  --latent_space_realign
  --think
  --max_samples "$SAMPLES"
  --generate_bs 4
  --temperature 0.6
  --top_p 0.95
  --max_new_tokens 2048
)

if [[ "$GPU_COUNT" -ge 2 ]]; then
  BACKEND_ARGS=(
    --use_vllm
    --use_second_HF_model
    --enable_prefix_caching
    --device cuda:0
    --device2 cuda:1
  )
else
  BACKEND_ARGS=(--device cuda:0)
fi

if [[ -f "$VECTOR_PATH" ]]; then
  echo "=== SEAL vector already exists — skipping extraction ==="
  echo "    ($VECTOR_PATH)"
elif [[ "${SKIP_VECTOR_EXTRACTION:-0}" == "1" ]]; then
  echo "=== SKIP_VECTOR_EXTRACTION=1 but vector missing: $VECTOR_PATH ==="
  exit 1
else
  echo "=== Extracting SEAL vector ==="
  $PYTHON scripts/extract_seal_vector.py \
    --model_name "$MODEL" \
    --layer "$SEAL_LAYER" \
    --n_per_class 80 \
    --max_scan 300 \
    --output "$VECTOR_PATH"
fi

echo "=== Eval only from here (~1–2 hrs on 1 GPU) ==="

echo "=== LatentMAS baseline (no SEAL) ==="
$PYTHON run.py --method latent_mas "${COMMON_ARGS[@]}" "${BACKEND_ARGS[@]}" \
  | tee results/pilot/latent_mas.json

echo "=== LatentMAS + SEAL ==="
$PYTHON run.py --method latent_mas "${COMMON_ARGS[@]}" "${BACKEND_ARGS[@]}" \
  --use_seal \
  --seal_vector_path "$VECTOR_PATH" \
  --seal_layer "$SEAL_LAYER" \
  --seal_coef "$SEAL_COEF" \
  --seal_mode latent \
  | tee results/pilot/latent_mas_seal.json

echo "=== Summary ==="
$PYTHON - <<'PY'
import json, re, glob, os
rows = []
for path in sorted(glob.glob("results/pilot/*.json")):
    text = open(path).read()
    matches = re.findall(r"\{[^{}]*\"accuracy\"[^{}]*\}", text)
    if not matches:
        continue
    data = json.loads(matches[-1])
    data["run"] = os.path.basename(path)
    rows.append(data)
for row in rows:
    print(row)
if len(rows) == 2:
    delta = rows[1]["accuracy"] - rows[0]["accuracy"]
    print(f"accuracy_delta={delta:+.4f}")
PY
