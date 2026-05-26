"""
PhysicsCFGSampleModel2 — inference wrapper for Physics CFG (trained version).

Requires a model trained with PhysicsPairedDataset (phys_flag condition).

At each denoising step runs the model TWICE:
    out_no_phys = model(x_t, t, y  with phys_flag=0, phys_uncond=False)
    out_phys    = model(x_t, t, y  with phys_flag=1, phys_uncond=False)

then blends:
    output = out_no_phys + phys_scale × (out_phys - out_no_phys)

phys_scale=0 → no physics (pure text CFG output)
phys_scale=1 → fully physics-conditioned
phys_scale∈(0,1) → smooth blend (recommended 0.5–0.7)
"""

import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr


class PhysicsCFGSampleModel2(nn.Module):
    """
    Parameters
    ----------
    model       : MDM (optionally already wrapped by ClassifierFreeSampleModel)
    phys_scale  : CFG blend weight; read from y['phys_scale'] if tensor
    """

    def __init__(self, model, phys_scale: float = 0.5):
        super().__init__()
        self.model      = model
        self.phys_scale = phys_scale

        # Forward inner model's sampling-loop attributes
        self.rot2xyz    = model.rot2xyz
        self.translation = model.translation
        self.njoints    = model.njoints
        self.nfeats     = model.nfeats
        self.data_rep   = model.data_rep
        self.cond_mode  = model.cond_mode
        self.encode_text = model.encode_text

    # ------------------------------------------------------------------
    def forward(self, x, timesteps, y=None):
        y = y or {}

        phys_scale = y.get('phys_scale', self.phys_scale)
        if isinstance(phys_scale, torch.Tensor):
            phys_scale = phys_scale.view(-1, 1, 1, 1)

        bs = x.shape[0]
        device = x.device

        # ── Run model with phys_flag=0 (no physics, MDM-style) ─────────
        y0 = {**y,
              'phys_flag':  torch.zeros(bs, dtype=torch.long, device=device),
              'phys_uncond': False}
        out_no_phys = self.model(x, timesteps, y0)

        if phys_scale == 0 or (isinstance(phys_scale, torch.Tensor)
                                and (phys_scale == 0).all()):
            return out_no_phys

        # ── Run model with phys_flag=1 (GT / physics) ──────────────────
        y1 = {**y,
              'phys_flag':  torch.ones(bs, dtype=torch.long, device=device),
              'phys_uncond': False}
        out_phys = self.model(x, timesteps, y1)

        # ── CFG blend ───────────────────────────────────────────────────
        return out_no_phys + phys_scale * (out_phys - out_no_phys)

    # ------------------------------------------------------------------
    def __getattr__(self, name, default=None):
        return wrapped_getattr(self, name, default=None)
