"""
physics_metrics.py — 量測生成動作的物理違規程度

從 results.npy 讀取關節位置，計算三個指標：
  floor_pen  : 腳踝/腳掌穿透地板的平均深度 (越小越好，理想=0)
  skate      : 接觸幀的腳掌速度 (越小越好，理想=0)
  float_h    : 接觸幀的腳掌離地高度 (越小越好，理想=0)

資料格式假設 results.npy 包含 hml_vec 已轉成 XYZ joint positions：
  motion : [N, n_joints, 3, T]  (由 sample/generate.py 的 rot2xyz 輸出)
"""

import numpy as np
import argparse
from pathlib import Path


# ── SMPL joint indices (22-joint HumanML3D ordering) ────────────────────────
# 0:Pelvis 1:L_Hip 2:R_Hip 3:Spine1 4:L_Knee 5:R_Knee 6:Spine2
# 7:L_Ankle 8:R_Ankle 9:Spine3 10:L_Foot 11:R_Foot ...
L_ANKLE, R_ANKLE = 7, 8
L_FOOT,  R_FOOT  = 10, 11
FOOT_JOINTS = [L_ANKLE, R_ANKLE, L_FOOT, R_FOOT]


def compute_physics_metrics(motion: np.ndarray, fps: float = 20.) -> dict:
    """
    motion : [n_joints, 3, T]  XYZ joint positions (已是世界座標)
    returns dict with scalar metrics (averaged over frames & samples)
    """
    # Y axis is the vertical (up) axis in HumanML3D
    foot_y = motion[FOOT_JOINTS, 1, :]   # [4, T]

    # ── 1. floor penetration ────────────────────────────────────────────────
    pen = np.maximum(0., -foot_y)          # > 0 only when below floor
    floor_pen = float(pen.mean())          # mean penetration depth

    # ── 2. foot skating  ────────────────────────────────────────────────────
    foot_pos = motion[FOOT_JOINTS, :, :]   # [4, 3, T]
    foot_vel = np.linalg.norm(
        foot_pos[:, :, 1:] - foot_pos[:, :, :-1], axis=1
    ) * fps   # [4, T-1], converted to m/s

    # Contact heuristic: foot is "grounded" when Y < threshold
    contact_thresh = 0.10                  # 10 cm above floor = contact
    contact_mask = (foot_y[:, :-1] < contact_thresh)  # [4, T-1]
    grounded_vel = foot_vel[contact_mask]
    skate = float(grounded_vel.mean()) if grounded_vel.size > 0 else 0.

    # ── 3. floating height  ─────────────────────────────────────────────────
    #   how high are feet when they "should" be grounded
    #   (use contact_mask from above, look at absolute Y)
    float_y = foot_y[:, :-1][contact_mask]
    float_h = float(float_y.mean()) if float_y.size > 0 else 0.

    return dict(floor_pen=floor_pen, skate=skate, float_h=float_h)


def evaluate_results_npy(npy_path: str, fps: float = 20.) -> dict:
    data   = np.load(npy_path, allow_pickle=True).item()
    motion = data['motion']          # [N, n_joints, 3, T]  (world XYZ)

    all_metrics = [compute_physics_metrics(motion[i], fps) for i in range(len(motion))]

    agg = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        agg[k]          = float(np.mean(vals))
        agg[k + '_std'] = float(np.std(vals))
    agg['n_samples'] = len(motion)
    return agg


def main():
    parser = argparse.ArgumentParser(description='Compute physics metrics on generated motions')
    parser.add_argument('results_npy', type=str, help='Path to results.npy produced by sample/generate.py')
    parser.add_argument('--fps', type=float, default=20., help='Frame rate (default 20 for humanml)')
    args = parser.parse_args()

    metrics = evaluate_results_npy(args.results_npy, args.fps)
    print(f"\n{'─'*45}")
    print(f"  Physics Metrics  ({metrics['n_samples']} samples)")
    print(f"{'─'*45}")
    print(f"  floor_pen  (↓)  {metrics['floor_pen']:.4f} ± {metrics['floor_pen_std']:.4f} m")
    print(f"  skate      (↓)  {metrics['skate']:.4f} ± {metrics['skate_std']:.4f} m/s")
    print(f"  float_h    (↓)  {metrics['float_h']:.4f} ± {metrics['float_h_std']:.4f} m")
    print(f"{'─'*45}\n")
    return metrics


if __name__ == '__main__':
    main()
