"""
VelocityCFGSampleModel — test-time velocity control via Classifier-Free Guidance.

Inference formula:
    out = out_uncond + vel_guidance_scale × (out_vel_target - out_uncond)

where out_uncond uses the null velocity embedding and out_vel_target uses
the user-specified target root velocity (in normalised HML_VEC units).
"""

import torch
import torch.nn as nn


class VelocityCFGSampleModel(nn.Module):
    def __init__(self, model, target_velocity: float, vel_guidance_scale: float = 2.0):
        """
        Args:
            model              : the base MDM model (already wrapped with ClassifierFreeSampleModel)
            target_velocity    : desired root velocity magnitude (normalised units)
            vel_guidance_scale : CFG scale (0 = unconditional, 1 = conditional, >1 = amplified)
        """
        super().__init__()
        self.model = model
        self.target_velocity = target_velocity
        self.vel_guidance_scale = vel_guidance_scale

    def forward(self, x, timesteps, y=None):
        bs = x.shape[0]
        device = x.device

        vel_target = torch.full((bs,), self.target_velocity, dtype=torch.float32, device=device)
        vel_null_mask = torch.ones(bs, dtype=torch.bool, device=device)   # all null
        vel_no_mask   = torch.zeros(bs, dtype=torch.bool, device=device)  # none null

        y_uncond = {**y, 'vel_cond': vel_target, 'vel_mask': vel_null_mask}
        y_cond   = {**y, 'vel_cond': vel_target, 'vel_mask': vel_no_mask}

        out_uncond = self.model(x, timesteps, y_uncond)
        out_cond   = self.model(x, timesteps, y_cond)

        return out_uncond + self.vel_guidance_scale * (out_cond - out_uncond)
