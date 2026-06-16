#!/usr/bin/env bash
# Native success-vector steering of the LatentMAS Judger decode (HF backend).
#
#   baseline  vs  +native-success (coef sweep)  vs  controls (reversed, random)
#
# The steering direction is extracted from LatentMAS's OWN Judger decode states
# (correct - wrong); see scripts/extract_native_success_vector.py. Injection uses
# the SEAL text hook (seal_mode=text) at the same layer the vector was read from.
#
# Detach-safe: run inside tmux.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: CUDA not available to torch."
  exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
TASK="${TASK:-gsm8k}"
SAMPLES="${SAMPLES:-500}"               # eval samples (test split)
EXTRACT_SAMPLES="${EXTRACT_SAMPLES:-500}"   # train problems for vector extraction
LATENT_STEPS="${LATENT_STEPS:-40}"
LAYER="${LAYER:-28}"
GEN_BS="${GEN_BS:-8}"
COEF_SWEEP="${COEF_SWEEP:-1 2 4 8}"     # success-vector dose
REV_COEF="${REV_COEF:-4}"               # reversed-vector control (applied as -REV_COEF)
RAND_COEF="${RAND_COEF:-4}"             # random-vector placebo control
DEVICE="${DEVICE:-cuda:0}"

VEC_DIR="${VEC_DIR:-artifacts/native_vectors/qwen3-14b}"
VEC_ALL="$VEC_DIR/success_layer${LAYER}_all.pt"
VEC_RAND="$VEC_DIR/random_layer${LAYER}.pt"

OUT="${OUT:-results/native_steer}"
mkdir -p "$OUT"

if [[ -z "${TMUX:-}" && "${ALLOW_NO_TMUX:-0}" != "1" ]]; then
  echo "ERROR: Not inside tmux — SSH/WiFi drops will kill this job."
  echo "  tmux new -s steer && cd $ROOT && bash scripts/run_native_steer_eval.sh"
  echo "Override (not recommended): ALLOW_NO_TMUX=1 bash scripts/run_native_steer_eval.sh"
  exit 1
fi

COMMON_ARGS=(
  --method latent_mas
  --model_name "$MODEL"
  --task "$TASK"
  --prompt sequential
  --latent_steps "$LATENT_STEPS"
  --latent_space_realign
  --think
  --max_samples "$SAMPLES"
  --generate_bs "$GEN_BS"
  --temperature 0.6
  --top_p 0.95
  --max_new_tokens 2048
  --device "$DEVICE"
)

# 1) Extract native success vector (skip if present).
if [[ -f "$VEC_ALL" ]]; then
  echo "=== Native vector exists — skipping extraction ($VEC_ALL) ==="
else
  echo "=== Extracting native success vector (${EXTRACT_SAMPLES} train problems, layer ${LAYER}) ==="
  $PYTHON scripts/extract_native_success_vector.py \
    --model_name "$MODEL" \
    --device "$DEVICE" \
    --layer "$LAYER" \
    --n_samples "$EXTRACT_SAMPLES" \
    --latent_steps "$LATENT_STEPS" \
    --out_dir "$VEC_DIR"
fi

# 2) Baseline (no steering).
echo "=== LatentMAS baseline (no steering) ==="
$PYTHON run.py "${COMMON_ARGS[@]}" | tee "$OUT/baseline.json"

# 3) Native success-vector sweep over coef (Judger decode steering).
for C in $COEF_SWEEP; do
  echo "=== +native success (layer=$LAYER, coef=$C, mode=text) ==="
  $PYTHON run.py "${COMMON_ARGS[@]}" \
    --use_seal --seal_mode text --seal_layer "$LAYER" \
    --seal_vector_path "$VEC_ALL" --seal_coef "$C" \
    | tee "$OUT/native_cv${C}.json"
done

# 4) Control: reversed direction (should HURT if the direction is causal).
echo "=== control: reversed success vector (coef=-$REV_COEF) ==="
$PYTHON run.py "${COMMON_ARGS[@]}" \
  --use_seal --seal_mode text --seal_layer "$LAYER" \
  --seal_vector_path "$VEC_ALL" --seal_coef "-$REV_COEF" \
  | tee "$OUT/control_reversed.json"

# 5) Control: norm-matched random vector (placebo — should do NOTHING).
echo "=== control: random vector (coef=$RAND_COEF) ==="
$PYTHON run.py "${COMMON_ARGS[@]}" \
  --use_seal --seal_mode text --seal_layer "$LAYER" \
  --seal_vector_path "$VEC_RAND" --seal_coef "$RAND_COEF" \
  | tee "$OUT/control_random.json"

# 6) Summary.
echo "=== Summary ==="
$PYTHON - <<'PY'
import json, re, glob, os
rows = []
for path in sorted(glob.glob("results/native_steer/*.json")):
    text = open(path).read()
    matches = re.findall(r"\{[^{}]*\"accuracy\"[^{}]*\}", text)
    if not matches:
        continue
    data = json.loads(matches[-1])
    rows.append({
        "run": os.path.basename(path),
        "accuracy": data.get("accuracy"),
        "correct": data.get("correct"),
        "seal_coef": data.get("seal_coef"),
        "seal_mode": data.get("seal_mode"),
    })
for r in rows:
    print(r)
PY
