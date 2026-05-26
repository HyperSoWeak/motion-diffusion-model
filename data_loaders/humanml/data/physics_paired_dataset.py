"""
PhysicsPairedDataset — standalone dataset for Physics CFG training.

不需要 GT motion。使用兩個 MDM-generated 的來源：
  phys_flag=0 → mdm_negatives/  (raw MDM output, 有 skating/floating 問題)
  phys_flag=1 → mdm_positives/  (physics-corrected MDM output, 物理改善版)

差值 (pos - neg) 代表物理修正方向，供 CFG 使用。

回傳格式與 Text2MotionDatasetV2 相同，讓 physics_t2m_collate 可以處理：
  (dummy, dummy, caption, sent_len, motion[T,263], length, tokens, None, phys_flag)
   b[0]   b[1]   b[2]     b[3]      b[4]            b[5]   b[6]   b[7]  b[8]
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join as pjoin


class PhysicsPairedDataset(Dataset):
    """
    Parameters
    ----------
    data_root   : path to HumanML3D dataset dir (contains train.txt, texts/, new_joint_vecs/)
    neg_dir     : path to MDM-generated negatives (generate_physics_negatives.py output)
    mean, std   : [263] numpy arrays for normalisation check (loaded from Mean.npy / Std.npy)
    split       : 'train' | 'val'
    max_frames  : truncate motions longer than this
    min_frames  : skip motions shorter than this
    pos_ratio   : probability of using GT (phys_flag=1)
    phys_mask_prob : probability of replacing phys_flag with null token (index 2)
    """

    PHYS_NULL = 2

    def __init__(self, data_root: str, neg_dir: str, pos_dir: str,
                 mean: np.ndarray, std: np.ndarray,
                 split: str = 'train',
                 max_frames: int = 196,
                 min_frames: int = 40,
                 pos_ratio: float = 0.5,
                 phys_mask_prob: float = 0.1):
        self.data_root = data_root
        self.neg_dir   = neg_dir
        self.pos_dir   = pos_dir
        self.mean      = mean
        self.std       = std
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.pos_ratio  = pos_ratio
        self.phys_mask_prob = phys_mask_prob

        # ── 讀 split IDs ──────────────────────────────────────────────
        split_file = pjoin(data_root, f'{split}.txt')
        all_ids = []
        seen = set()
        with open(split_file) as f:
            for line in f:
                sid = line.strip().lstrip('M')
                if sid and sid not in seen:
                    seen.add(sid)
                    all_ids.append(sid)

        # ── 過濾：neg 和 pos 都必須存在，且長度合法 ─────────────────────
        self.samples = []   # list of (sample_id, caption, length)
        for sid in all_ids:
            neg_path = pjoin(neg_dir, f'{sid}.npy')
            pos_path = pjoin(pos_dir, f'{sid}.npy')
            txt_path = pjoin(data_root, 'texts', f'{sid}.txt')

            if not (os.path.exists(neg_path) and
                    os.path.exists(pos_path) and
                    os.path.exists(txt_path)):
                continue

            neg = np.load(neg_path)
            T   = min(neg.shape[0], max_frames)
            if T < min_frames:
                continue

            caption = self._read_caption(txt_path)
            if caption is None:
                continue

            self.samples.append((sid, caption, T))

        print(f'[PhysicsPairedDataset] {split}: {len(self.samples)} valid samples')

        # ── 建立與 HumanML3D 相容的 attributes ──────────────────────
        self.name_list = [s[0] for s in self.samples]
        # mean_gpu / std_gpu 供 training_loop 的 dataset.mean_gpu 存取
        self.mean_gpu = None   # 由外部在 train_mdm.py 設置
        self.std_gpu  = None

        # 供 t2m_dataset.opt.joints_num 存取（如果有 eval）
        class _Opt: joints_num = 22
        class _T2M:
            opt = _Opt()
            def inv_transform(self, x): return x * std + mean
        self.t2m_dataset = _T2M()
        self.opt = _Opt()

    # ------------------------------------------------------------------
    @staticmethod
    def _read_caption(txt_path: str) -> str | None:
        with open(txt_path) as f:
            line = f.readline().strip()
        if not line:
            return None
        return line.split('#')[0].strip()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, caption, T = self.samples[idx]

        # ── 選 physics-corrected (flag=1) 或 raw MDM (flag=0) ────────
        use_pos = (random.random() < self.pos_ratio)
        if use_pos:
            motion = np.load(pjoin(self.pos_dir, f'{sid}.npy'))   # physics-corrected
            phys_flag = 1
        else:
            motion = np.load(pjoin(self.neg_dir, f'{sid}.npy'))   # raw MDM
            phys_flag = 0

        # Trim / pad to T frames
        motion = motion[:T]   # [T, 263]
        if motion.shape[0] < T:
            pad = np.zeros((T - motion.shape[0], 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        # ── phys_flag masking ─────────────────────────────────────────
        if random.random() < self.phys_mask_prob:
            phys_flag = self.PHYS_NULL

        # ── 回傳格式與 Text2MotionDatasetV2 相同 (b[0..8]) ──────────
        # b[0]=dummy, b[1]=dummy, b[2]=caption, b[3]=sent_len,
        # b[4]=motion[T,263], b[5]=length, b[6]=tokens, b[7]=None, b[8]=phys_flag
        tokens  = '_'.join(caption.split())
        sent_len = len(caption.split())
        dummy    = np.zeros(1, dtype=np.float32)

        return dummy, dummy, caption, sent_len, motion.astype(np.float32), T, tokens, None, phys_flag
