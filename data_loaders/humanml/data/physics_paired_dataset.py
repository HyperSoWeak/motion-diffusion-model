"""
PhysicsPairedDataset — standalone dataset for Physics CFG training.

phys_flag=0 → mdm_negatives/  (raw MDM output, normalized HML_VEC)
phys_flag=1 → new_joint_vecs/ (GT HumanML3D, unnormalized → normalized here)
phys_flag=2 → null token (phys_mask_prob, for CFG unconditional training)

The GT data is the correct positive: natural AND physically grounded.

Return format matches Text2MotionDatasetV2:
  (dummy, dummy, caption, sent_len, motion[T,263], length, tokens, None, phys_flag)
   b[0]   b[1]   b[2]     b[3]      b[4]            b[5]   b[6]   b[7]  b[8]
"""

import os
import random
import numpy as np
from torch.utils.data import Dataset
from os.path import join as pjoin


class PhysicsPairedDataset(Dataset):
    """
    Parameters
    ----------
    data_root      : HumanML3D dir (contains train.txt, texts/, new_joint_vecs/)
    neg_dir        : MDM-generated negatives (normalized HML_VEC, from generate_physics_negatives.py)
    pos_dir        : IGNORED when gt_dir is set; kept for backward compatibility
    mean, std      : [263] arrays for GT normalization (Mean.npy / Std.npy)
    gt_dir         : path to new_joint_vecs/ — if set, used as phys_flag=1 positives
    split          : 'train' | 'val'
    max_frames     : truncate motions longer than this
    min_frames     : skip motions shorter than this
    pos_ratio      : fraction of samples using phys_flag=1
    phys_mask_prob : probability of replacing phys_flag with null token (2)
    """

    PHYS_NULL = 2

    def __init__(self, data_root: str, neg_dir: str, pos_dir: str,
                 mean: np.ndarray, std: np.ndarray,
                 gt_dir: str = '',
                 split: str = 'train',
                 max_frames: int = 196,
                 min_frames: int = 40,
                 pos_ratio: float = 0.5,
                 phys_mask_prob: float = 0.1):
        self.neg_dir   = neg_dir
        self.pos_dir   = pos_dir          # Adam-corrected positives (legacy)
        self.gt_dir    = gt_dir           # GT new_joint_vecs/ (preferred)
        self.mean      = mean
        self.std       = std
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.pos_ratio  = pos_ratio
        self.phys_mask_prob = phys_mask_prob

        use_gt = bool(gt_dir)

        # ── Read split IDs ────────────────────────────────────────────
        split_file = pjoin(data_root, f'{split}.txt')
        all_ids = []
        seen = set()
        with open(split_file) as f:
            for line in f:
                sid = line.strip().lstrip('M')
                if sid and sid not in seen:
                    seen.add(sid)
                    all_ids.append(sid)

        # ── Filter: neg + pos/gt must exist, length valid ─────────────
        self.samples = []   # (sample_id, caption, T_gt, T_neg)
        for sid in all_ids:
            neg_path = pjoin(neg_dir, f'{sid}.npy')
            txt_path = pjoin(data_root, 'texts', f'{sid}.txt')

            if use_gt:
                pos_path = pjoin(gt_dir, f'{sid}.npy')
            else:
                pos_path = pjoin(pos_dir, f'{sid}.npy')

            if not (os.path.exists(neg_path) and
                    os.path.exists(pos_path) and
                    os.path.exists(txt_path)):
                continue

            neg = np.load(neg_path)
            T_neg = min(neg.shape[0], max_frames)
            if T_neg < min_frames:
                continue

            if use_gt:
                gt = np.load(pos_path)
                T_gt = min(gt.shape[0], max_frames)
                if T_gt < min_frames:
                    continue
            else:
                T_gt = T_neg   # Adam positives have same length as neg

            caption = self._read_caption(txt_path)
            if caption is None:
                continue

            self.samples.append((sid, caption, T_gt, T_neg))

        mode = 'GT' if use_gt else 'Adam-positives'
        print(f'[PhysicsPairedDataset] {split} ({mode}): {len(self.samples)} valid samples')

        # ── Attributes for HumanML3D compatibility ────────────────────
        self.name_list = [s[0] for s in self.samples]
        self.mean_gpu = None
        self.std_gpu  = None

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
        sid, caption, T_gt, T_neg = self.samples[idx]

        use_pos = (random.random() < self.pos_ratio)
        if use_pos:
            if self.gt_dir:
                # GT: unnormalized → normalize
                raw = np.load(pjoin(self.gt_dir, f'{sid}.npy'))[:T_gt]  # [T, 263]
                motion = ((raw - self.mean) / self.std).astype(np.float32)
                T = T_gt
            else:
                # Adam positives: already normalized
                motion = np.load(pjoin(self.pos_dir, f'{sid}.npy'))[:T_neg].astype(np.float32)
                T = T_neg
            phys_flag = 1
        else:
            # MDM negatives: already normalized
            motion = np.load(pjoin(self.neg_dir, f'{sid}.npy'))[:T_neg].astype(np.float32)
            T = T_neg
            phys_flag = 0

        # Pad if truncated
        if motion.shape[0] < T:
            pad = np.zeros((T - motion.shape[0], 263), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        if random.random() < self.phys_mask_prob:
            phys_flag = self.PHYS_NULL

        tokens   = '_'.join(caption.split())
        sent_len = len(caption.split())
        dummy    = np.zeros(1, dtype=np.float32)

        return dummy, dummy, caption, sent_len, motion, T, tokens, None, phys_flag
