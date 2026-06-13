#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/exp_logs save/physics_ablate_5ep save/physics_ablate32

BASE="save/humanml_enc_512_50steps/model000750000.pt"
PROMPTS="reports/physics_eval_prompts_32.txt"
LAM_PHYS=0.01
STEPS=3068

get_latest_checkpoint() {
  local dir="$1"
  ls "$dir"/model*.pt 2>/dev/null | sort -V | tail -n 1
}

generate_eval() {
  local name="$1"
  local ckpt="$2"
  local out="save/physics_ablate32/${name}_samples"

  python -m sample.generate \
    --model_path "$ckpt" \
    --dataset humanml \
    --input_text "$PROMPTS" \
    --output_dir "$out" \
    2>&1 | tee "reports/exp_logs/gen_${name}.log"
    # --num_repetitions 2 \
    # --guidance_param 2.5 \
    # --skip_visualization \

  python eval/eval_physics.py --results_path "${out}/results.npy"
}

train_one() {
  local name="$1"
  local g="$2"
  local f="$3"
  local s="$4"
  local save_dir="save/physics_ablate_5ep/${name}"
  local log="reports/exp_logs/train_${name}.log"

  python -m train.train_mdm --overwrite --save_dir "$save_dir" \
    --resume_checkpoint "$BASE" --dataset humanml --diffusion_steps 50 \
    --batch_size 32 --lr 1e-5 --num_steps "$STEPS" --save_interval 1000 --log_interval 50 \
    --lambda_phys "$LAM_PHYS" \
    --lambda_phys_ground "$g" --lambda_phys_foot "$f" --lambda_phys_smooth "$s" \
    > "$log" 2>&1
  
  local ckpt
  ckpt="$(get_latest_checkpoint "$save_dir")"
  echo "$ckpt"
}

# 名稱 = ablate_[g][f][s][r]；本輪 r 固定為 0
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

# --- 7 組 finetune + generate + eval ---
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  metrics="save/physics_ablate32/${name}_samples/physics_metrics.json"
  if [ -f "$metrics" ]; then
    echo "[$name] already done, skip"
    continue
  fi
  ckpt="$(train_one "$name" "${G[$i]}" "${F[$i]}" "${S[$i]}")"
  echo "[$name] checkpoint: $ckpt"
  generate_eval "$name" "$ckpt"
done

# --- ablate_0000：baseline（不 finetune）---
generate_eval "ablate_0000" "$BASE"

echo "Done. Metrics under save/physics_ablate32/*_samples/physics_metrics.json"
