#!/usr/bin/env bash
# Full ablation: finetune (7 runs) + generate + eval + baseline.
# Run names: ablate_[g][f][s][r], this round r=0.
set -euo pipefail

cd "$(dirname "$0")"

# === Change RUN_TAG when starting a fresh full rerun ===
RUN_TAG="${RUN_TAG:-v3}"
SAVE_TRAIN="${SAVE_TRAIN:-save/physics_ablate_5ep_${RUN_TAG}}"
SAVE_EVAL="${SAVE_EVAL:-save/physics_ablate32_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-reports/exp_logs/${RUN_TAG}}"

BASE="${BASE:-save/humanml_enc_512_50steps/model000750000.pt}"
PROMPTS="${PROMPTS:-reports/physics_eval_prompts_32.txt}"
LAM_PHYS="${LAM_PHYS:-0.01}"
STEPS="${STEPS:-3068}"

SEED="${SEED:-10}"
NUM_REPETITIONS="${NUM_REPETITIONS:-2}"
GUIDANCE_PARAM="${GUIDANCE_PARAM:-2.5}"
DEVICE="${DEVICE:-0}"

mkdir -p "$LOG_DIR" "$SAVE_TRAIN" "$SAVE_EVAL"

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

train_one() {
  local name="$1"
  local g="$2"
  local f="$3"
  local s="$4"
  local save_dir="${SAVE_TRAIN}/${name}"
  local log="${LOG_DIR}/train_${name}.log"

  echo "[$name] train -> $save_dir" >&2

  python -m train.train_mdm --overwrite --save_dir "$save_dir" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume_checkpoint "$BASE" --dataset humanml --diffusion_steps 50 \
    --batch_size 32 --lr 1e-5 --num_steps "$STEPS" --save_interval 1000 --log_interval 50 \
    --lambda_phys "$LAM_PHYS" \
    --lambda_phys_ground "$g" --lambda_phys_foot "$f" --lambda_phys_smooth "$s" \
    > "$log" 2>&1

  local ckpt
  ckpt="$(get_latest_checkpoint "$save_dir")"
  if [[ -z "$ckpt" ]]; then
    echo "ERROR: no checkpoint in $save_dir" >&2
    exit 1
  fi
  echo "$ckpt"
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
declare -a G=(1.0 0 0 1.0 1.0 0 1.0)
declare -a F=(0 1.0 0 1.0 0 1.0 1.0)
declare -a S=(0 0 0.05 0 0.05 0.05 0.05)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  metrics="${SAVE_EVAL}/${name}_samples/physics_metrics.json"
  if [[ -f "$metrics" ]]; then
    echo "[$name] already done, skip"
    continue
  fi

  ckpt="$(train_one "$name" "${G[$i]}" "${F[$i]}" "${S[$i]}")"
  echo "[$name] checkpoint: $ckpt"
  generate_eval "$name" "$ckpt"
done

metrics="${SAVE_EVAL}/ablate_0000_samples/physics_metrics.json"
if [[ -f "$metrics" ]]; then
  echo "[ablate_0000] already done, skip"
else
  generate_eval "ablate_0000" "$BASE"
fi

echo "Done."
echo "  checkpoints: ${SAVE_TRAIN}/"
echo "  metrics:     ${SAVE_EVAL}/*_samples/physics_metrics.json"
echo "Collect report:"
echo "  python collect_metrics.py --eval-root ${SAVE_EVAL} \\"
echo "    --out-json reports/physics_ablation_study_${RUN_TAG}_metrics.json \\"
echo "    --out-md reports/physics_ablation_study_${RUN_TAG}_report.md"
