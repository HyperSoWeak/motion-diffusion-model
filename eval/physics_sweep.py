"""
physics_sweep.py — 自動 sweep phys_scale（Physics CFG），量測 trade-off

用法：
    python -m eval.physics_sweep \
        --model_path save/humanml_enc_512_50steps/model000750000.pt \
        --text_prompt "a person walks forward." \
        --scales 0 0.1 0.2 0.3 0.5 0.7 1.0 \
        --num_samples 6 --device 0

輸出：
  - 每個 scale 的物理指標表格（列印 + 存 CSV）
  - sweep_results.csv 可用 Excel/pandas 畫圖
"""

import argparse
import subprocess
import sys
import os
import csv
import json
from pathlib import Path

# 讓 eval/physics_metrics 可以 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.physics_metrics import evaluate_results_npy


def run_generate(model_path, text_prompt, out_dir, scale,
                 num_samples, num_reps, device,
                 guidance_param, floor_w, skate_w, float_w,
                 phys_optim_steps, phys_lr):
    """呼叫 sample.generate，回傳 results.npy 路徑"""
    cmd = [
        sys.executable, '-m', 'sample.generate',
        '--model_path', model_path,
        '--text_prompt', text_prompt,
        '--output_dir', out_dir,
        '--num_samples', str(num_samples),
        '--num_repetitions', str(num_reps),
        '--device', str(device),
        '--guidance_param', str(guidance_param),
        '--phys_scale', str(scale),
        '--physics_floor_weight', str(floor_w),
        '--physics_skate_weight', str(skate_w),
        '--physics_float_weight', str(float_w),
        '--phys_optim_steps', str(phys_optim_steps),
        '--phys_lr', str(phys_lr),
        '--seed', '42',
    ]
    print(f'\n[sweep] scale={scale:.1f}  →  {out_dir}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('[sweep] FAILED:\n', result.stderr[-2000:])
        return None
    npy = os.path.join(out_dir, 'results.npy')
    return npy if os.path.exists(npy) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--text_prompt', default='a person walks forward.')
    parser.add_argument('--scales', nargs='+', type=float,
                        default=[0., 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                        help='List of phys_scale (Physics CFG) values to sweep')
    parser.add_argument('--num_samples', type=int, default=6)
    parser.add_argument('--num_reps', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--guidance_param', type=float, default=2.5)
    parser.add_argument('--floor_w', type=float, default=10.)
    parser.add_argument('--skate_w', type=float, default=5.)
    parser.add_argument('--float_w', type=float, default=10.)
    parser.add_argument('--phys_optim_steps', type=int, default=3,
                        help='Adam steps per denoising step for physics CFG')
    parser.add_argument('--phys_lr', type=float, default=0.05,
                        help='Adam lr for physics CFG optimisation')
    parser.add_argument('--output_dir', default='save/physics_sweep_cfg')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'sweep_results.csv')

    rows = []
    for scale in args.scales:
        out_dir = os.path.join(args.output_dir, f'scale_{scale:.1f}')
        npy = run_generate(
            model_path=args.model_path,
            text_prompt=args.text_prompt,
            out_dir=out_dir,
            scale=scale,
            num_samples=args.num_samples,
            num_reps=args.num_reps,
            device=args.device,
            guidance_param=args.guidance_param,
            floor_w=args.floor_w,
            skate_w=args.skate_w,
            float_w=args.float_w,
            phys_optim_steps=args.phys_optim_steps,
            phys_lr=args.phys_lr,
        )
        if npy is None:
            print(f'  [skip] scale={scale}  (generation failed)')
            continue

        metrics = evaluate_results_npy(npy)
        row = {'scale': scale, **metrics}
        rows.append(row)

        print(f'  floor_pen={metrics["floor_pen"]:.4f}  '
              f'skate={metrics["skate"]:.4f}  '
              f'float_h={metrics["float_h"]:.4f}')

    # ── Save CSV ──
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n[sweep] Results saved to {csv_path}')

    # ── Summary table ──
    print(f'\n{"scale":>8}  {"floor_pen":>10}  {"skate":>10}  {"float_h":>10}')
    print('─' * 46)
    for r in rows:
        print(f'{r["scale"]:>8.1f}  {r["floor_pen"]:>10.4f}  {r["skate"]:>10.4f}  {r["float_h"]:>10.4f}')

    # ── Simple recommendation ──
    if rows:
        # pick the scale where improvement in floor_pen+skate is diminishing
        baseline = rows[0]  # scale=0
        print('\n[sweep] Improvement vs. baseline (scale=0):')
        print(f'{"scale":>8}  {"Δfloor_pen":>12}  {"Δskate":>12}  {"Δfloat_h":>12}')
        print('─' * 50)
        for r in rows[1:]:
            dp = r['floor_pen'] - baseline['floor_pen']
            ds = r['skate']     - baseline['skate']
            dh = r['float_h']   - baseline['float_h']
            print(f'{r["scale"]:>8.1f}  {dp:>+12.4f}  {ds:>+12.4f}  {dh:>+12.4f}')


if __name__ == '__main__':
    main()
