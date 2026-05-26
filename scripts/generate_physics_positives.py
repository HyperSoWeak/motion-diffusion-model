"""
generate_physics_positives.py
對 dataset/mdm_negatives/ 裡的每筆 MDM 輸出做物理修正（Adam on hml_vec），
存成 dataset/mdm_positives/ 作為 Physics CFG training 的 phys_flag=1 樣本。

不需要 GT motion，只需要 mdm_negatives/ 和 dataset mean/std。

用法：
    python -m scripts.generate_physics_positives \
        --neg_dir dataset/mdm_negatives \
        --out_dir dataset/mdm_positives \
        --optim_steps 100 \
        --lr 0.05 \
        --device 0

Resume：已存在的檔案自動跳過。
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── Physics energy（與 diffusion/physics_guidance.py 相同） ──────────────────

FOOT_RIC_Y_IDXS = [23, 32, 26, 35]
ROOT_Y_IDX      = 3
FC_START, FC_END = 259, 263


def compute_energy(x: torch.Tensor,
                   floor_w=10., skate_w=5., float_w=10.) -> torch.Tensor:
    """x : [1, 263, 1, T]  denormalised hml_vec"""
    root_y    = x[:, ROOT_Y_IDX, 0, :]                        # [1, T]
    foot_abs_y = torch.stack(
        [root_y + x[:, idx, 0, :] for idx in FOOT_RIC_Y_IDXS], dim=1
    )                                                          # [1, 4, T]

    fc_soft = torch.sigmoid(x[:, FC_START:FC_END, 0, :] * 5.) # [1, 4, T]

    floor_e = F.relu(-foot_abs_y).pow(2).sum()
    foot_vel = (foot_abs_y[:, :, 1:] - foot_abs_y[:, :, :-1]).pow(2)
    skate_e  = (foot_vel * fc_soft[:, :, :-1]).sum()
    float_e  = (foot_abs_y.pow(2) * fc_soft).sum()

    return floor_w * floor_e + skate_w * skate_e + float_w * float_e


def physics_correct(arr_norm: np.ndarray,
                    mean: np.ndarray, std: np.ndarray,
                    optim_steps: int, lr: float,
                    device: str,
                    floor_w=10., skate_w=5., float_w=10.) -> np.ndarray:
    """
    arr_norm : [T, 263] normalised hml_vec
    returns  : [T, 263] physics-corrected normalised hml_vec
    """
    # [T, 263] → [1, 263, 1, T] normalised
    x_norm = torch.tensor(arr_norm.T, dtype=torch.float32, device=device
                          ).unsqueeze(0).unsqueeze(2)          # [1, 263, 1, T]

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device
                          ).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device
                          ).view(1, -1, 1, 1)

    x = x_norm.clone().requires_grad_(True)
    optim = torch.optim.Adam([x], lr=lr)

    with torch.enable_grad():
        for _ in range(optim_steps):
            optim.zero_grad()
            x_denorm = x * std_t + mean_t
            loss = compute_energy(x_denorm, floor_w, skate_w, float_w)
            loss.backward()
            optim.step()

    # [1, 263, 1, T] → [T, 263]
    out = x.detach().cpu()[0, :, 0, :].T.numpy()
    return out.astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_dir',  default='dataset/mdm_negatives')
    parser.add_argument('--out_dir',  default='dataset/mdm_positives')
    parser.add_argument('--mean_path', default='dataset/HumanML3D/Mean.npy')
    parser.add_argument('--std_path',  default='dataset/HumanML3D/Std.npy')
    parser.add_argument('--optim_steps', type=int, default=100)
    parser.add_argument('--lr',          type=float, default=0.05)
    parser.add_argument('--floor_w',     type=float, default=10.)
    parser.add_argument('--skate_w',     type=float, default=5.)
    parser.add_argument('--float_w',     type=float, default=10.)
    parser.add_argument('--device',      type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}   optim_steps={args.optim_steps}   lr={args.lr}')

    mean = np.load(args.mean_path)
    std  = np.load(args.std_path)

    # 讀 meta.json 取所有 sample IDs
    meta_path = os.path.join(args.neg_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    sample_ids = sorted(meta.keys())
    todo = [sid for sid in sample_ids
            if not os.path.exists(os.path.join(args.out_dir, f'{sid}.npy'))]
    print(f'Total: {len(sample_ids)}   Remaining: {len(todo)}')

    for sid in tqdm(todo, desc='Physics correcting'):
        neg_path = os.path.join(args.neg_dir, f'{sid}.npy')
        arr = np.load(neg_path)   # [T, 263]

        corrected = physics_correct(
            arr, mean, std,
            optim_steps=args.optim_steps,
            lr=args.lr,
            device=device,
            floor_w=args.floor_w,
            skate_w=args.skate_w,
            float_w=args.float_w,
        )

        out_path = os.path.join(args.out_dir, f'{sid}.npy')
        np.save(out_path, corrected)

    print(f'\nDone. Positives saved to {args.out_dir}')


if __name__ == '__main__':
    main()
