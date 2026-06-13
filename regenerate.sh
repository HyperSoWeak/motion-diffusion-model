#!/usr/bin/env bash
# Re-run sample.generate + eval_physics only (no finetune).
# Bit order in run names: [g][f][s][r]
set -euo pipefail

cd "$(dirname "$0")"

SAVE_EVAL="${SAVE_EVAL:-save/physics_ablate32_v2}"
LOG_DIR="${LOG_DIR:-reports/exp_logs/v2}"
TRAIN_ROOT="${TRAIN_ROOT:-save/physics_ablate_5ep}"
BASE="${BASE:-save/humanml_enc_512_50steps/model000750000.pt}"
PROMPTS="${PROMPTS:-reports/physics_eval_prompts_32.txt}"

SEED="${SEED:-10}"
NUM_REPETITIONS="${NUM_REPETITIONS:-2}"
GUIDANCE_PARAM="${GUIDANCE_PARAM:-2.5}"
DEVICE="${DEVICE:-0}"

mkdir -p "$LOG_DIR" "$SAVE_EVAL"

get_latest_checkpoint() {
  local dir="$1"
  ls "$dir"/model*.pt 2>/dev/null | sort -V | tail -n 1
}

generate_eval() {
  local name="$1"
  local ckpt="$2"
  local out="${SAVE_EVAL}/${name}_samples"

  echo "[$name] generate from: $ckpt"
  echo "[$name] output: $out"

  python -m sample.generate \
    --device "$DEVICE" \
    --model_path "$ckpt" \
    --dataset humanml \
    --input_text "$PROMPTS" \
    --num_repetitions "$NUM_REPETITIONS" \
    --guidance_param "$GUIDANCE_PARAM" \
    --seed "$SEED" \
    --skip_visualization \
    --output_dir "$out" \
    2>&1 | tee "${LOG_DIR}/gen_${name}.log"

  python eval/eval_physics.py --results_path "${out}/results.npy"
}

declare -a NAMES=(
  "ablate_1000"
  "ablate_0100"
  "ablate_0010"
  "ablate_1100"
  "ablate_1010"
  "ablate_0110"
  "ablate_1110"
)

for name in "${NAMES[@]}"; do
  metrics="${SAVE_EVAL}/${name}_samples/physics_metrics.json"
  if [[ -f "$metrics" ]]; then
    echo "[$name] already done, skip"
    continue
  fi

  ckpt="$(get_latest_checkpoint "${TRAIN_ROOT}/${name}")"
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: no checkpoint in ${TRAIN_ROOT}/${name}" >&2
    exit 1
  fi

  generate_eval "$name" "$ckpt"
done

metrics="${SAVE_EVAL}/ablate_0000_samples/physics_metrics.json"
if [[ -f "$metrics" ]]; then
  echo "[ablate_0000] already done, skip"
else
  generate_eval "ablate_0000" "$BASE"
fi

echo "Done. Metrics under ${SAVE_EVAL}/*_samples/physics_metrics.json"
echo "Collect report:"
echo "  python collect_metrics.py --eval-root ${SAVE_EVAL} \\"
echo "    --out-json reports/physics_ablation_study_v2_metrics.json \\"
echo "    --out-md reports/physics_ablation_study_v2_report.md"
