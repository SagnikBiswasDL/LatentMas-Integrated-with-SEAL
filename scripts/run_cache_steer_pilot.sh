#!/usr/bin/env bash
# Cache-steering A/B + coefficient sweep for LatentMAS on GSM8K (HF backend).
#
#   LatentMAS (baseline)  vs  LatentMAS + cache steering (sweep c_v)
#
# Cache steering needs the HF backend (it edits the KV cache directly), so this
# script never uses vLLM. Detach-safe: run inside tmux.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

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
SAMPLES="${SAMPLES:-200}"
LATENT_STEPS="${LATENT_STEPS:-40}"
GEN_BS="${GEN_BS:-8}"                 # larger batch size to speed up runtime
CK="${CK:-0.0}"                       # GSM8K paper default: key coef = 0
CV_SWEEP="${CV_SWEEP:-1 3 6}"         # value coefficient sweep
POSITIONS="${POSITIONS:-last_n}"
LAST_N="${LAST_N:-$LATENT_STEPS}"
N_PAIRS="${N_PAIRS:-200}"
N_ICL="${N_ICL:-5}"
VECTOR_PATH="${VECTOR_PATH:-artifacts/cache_steer_vectors/qwen3-14b/${TASK}_kv.pt}"

OUT="${OUT:-results/cache_steer}"
mkdir -p "$OUT"

if [[ -z "${TMUX:-}" && "${ALLOW_NO_TMUX:-0}" != "1" ]]; then
  echo "ERROR: Not inside tmux — SSH/WiFi drops will kill this job."
  echo "  tmux new -s cachesteer && cd $ROOT && bash scripts/run_cache_steer_pilot.sh"
  echo "Override (not recommended): ALLOW_NO_TMUX=1 bash scripts/run_cache_steer_pilot.sh"
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
  --device cuda:0
)

# 1) Extract cache-steering vectors (skip if present).
if [[ -f "$VECTOR_PATH" ]]; then
  echo "=== Cache-steering vector exists — skipping extraction ($VECTOR_PATH) ==="
else
  echo "=== Extracting cache-steering vectors (${N_PAIRS} pairs, ${N_ICL}-shot) ==="
  $PYTHON scripts/extract_cache_steering_vectors.py \
    --model_name "$MODEL" \
    --n_pairs "$N_PAIRS" \
    --n_icl "$N_ICL" \
    --output "$VECTOR_PATH"
fi

# 2) Baseline (no steering).
echo "=== LatentMAS baseline (no cache steering) ==="
$PYTHON run.py "${COMMON_ARGS[@]}" | tee "$OUT/baseline.json"

# 3) Cache-steering sweep over c_v.
for CV in $CV_SWEEP; do
  echo "=== LatentMAS + cache steering (c_k=$CK, c_v=$CV, pos=$POSITIONS) ==="
  $PYTHON run.py "${COMMON_ARGS[@]}" \
    --use_cache_steer \
    --cache_steer_vector_path "$VECTOR_PATH" \
    --cache_steer_ck "$CK" \
    --cache_steer_cv "$CV" \
    --cache_steer_positions "$POSITIONS" \
    --cache_steer_last_n "$LAST_N" \
    | tee "$OUT/cache_steer_cv${CV}.json"
done

# 4) Summary.
echo "=== Summary ==="
$PYTHON - <<'PY'
import json, re, glob, os
rows = []
for path in sorted(glob.glob("results/cache_steer/*.json")):
    text = open(path).read()
    matches = re.findall(r"\{[^{}]*\"accuracy\"[^{}]*\}", text)
    if not matches:
        continue
    data = json.loads(matches[-1])
    data["run"] = os.path.basename(path)
    rows.append({k: data.get(k) for k in ("run", "accuracy", "correct", "cache_steer_cv", "total_time_sec")})
for r in rows:
    print(r)
PY
