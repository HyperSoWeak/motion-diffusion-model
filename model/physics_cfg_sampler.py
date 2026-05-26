"""
Physics Classifier-Free Guidance sampler for MDM.

Standard text CFG interpolates two model outputs:
    out = out_uncond + s_text × (out_cond_text - out_uncond)

This module adds a second CFG stage for physics:
    out_phys  = argmin_{pred_x0} E_physics(pred_x0)   (a few gradient steps)
    out_final = out_text + s_phys × (out_phys - out_text)

where out_text already contains text-CFG.  Combined:
    out_final = out_uncond
              + s_text × (out_cond_text - out_uncond)   # text guidance
              + s_phys × (out_phys      - out_text)     # physics guidance

Key difference from classifier guidance (previous approach):
  - Classifier guidance:  gradient w.r.t. x_t (noisy input)
                          → unstable, NaN risk, hard to interpret scale
  - CFG (this file):      gradient w.r.t. pred_x0 (clean prediction)
                          → stable, scale ∈ [0, 1] has clear meaning,
                            no change to the diffusion trajectory structure

Physics energy is the same floor / skating / floating energy from
diffusion/physics_guidance.py — no new loss definition needed.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr


class PhysicsCFGSampleModel(nn.Module):
    """
    Wraps an MDM model (optionally already wrapped by ClassifierFreeSampleModel
    for text CFG) and adds a physics CFG stage on top.

    Parameters
    ----------
    model            : the inner model (text-CFG-wrapped or plain MDM)
    physics_energy   : PhysicsGuidanceFn instance (only .compute_energy is used)
    phys_optim_steps : gradient-descent steps on pred_x0 per denoising step
    phys_lr          : Adam lr for the physics optimisation
    """

    def __init__(self, model, physics_energy, phys_optim_steps: int = 3, phys_lr: float = 0.05):
        super().__init__()
        self.model            = model
        self.physics_energy   = physics_energy
        self.phys_optim_steps = phys_optim_steps
        self.phys_lr          = phys_lr

        # Expose inner model attributes expected by the sampling loop
        self.rot2xyz   = model.rot2xyz
        self.translation = model.translation
        self.njoints   = model.njoints
        self.nfeats    = model.nfeats
        self.data_rep  = model.data_rep
        self.cond_mode = model.cond_mode
        self.encode_text = model.encode_text

    # ------------------------------------------------------------------
    def _physics_correct(self, pred_x0: torch.Tensor) -> torch.Tensor:
        """
        Optimise pred_x0 toward lower physics energy using Adam.

        This is the "physics-conditioned" output in the CFG formula.
        We use the SAME energy function as before (floor penetration +
        foot skating + floating) — no new loss definition.

        pred_x0 : [bs, 263, 1, nframes]  normalised hml_vec
        returns  : physics-improved pred_x0 (same shape, detached)
        """
        x = pred_x0.detach().clone().requires_grad_(True)
        optim = torch.optim.Adam([x], lr=self.phys_lr)

        with torch.enable_grad():          # override torch.no_grad() from p_sample
            for _ in range(self.phys_optim_steps):
                optim.zero_grad()
                loss = self.physics_energy.compute_energy(x)
                loss.backward()
                optim.step()

        return x.detach()

    # ------------------------------------------------------------------
    def forward(self, x, timesteps, y=None):
        """
        CFG forward pass.

        Step 1 — text-guided pred_x0 (inner model, may already apply text CFG):
            out_text = model(x_t, t, y)

        Step 2 — physics-conditioned pred_x0:
            out_phys = argmin_{pred_x0} E_physics  (gradient descent, warm-started at out_text)

        Step 3 — CFG interpolation:
            out = out_text + phys_scale × (out_phys - out_text)

        phys_scale=0 → pure text guidance (no physics)
        phys_scale=1 → fully physics-corrected output
        phys_scale∈(0,1) → smooth blend (recommended: 0.3–0.7)
        """
        # ── Step 1: text-guided prediction ─────────────────────────────
        out_text = self.model(x, timesteps, y)   # [bs, 263, 1, nframes]

        # ── Step 2: physics correction (same energy, optimise pred_x0) ─
        out_phys = self._physics_correct(out_text)

        # ── Step 3: CFG blend ──────────────────────────────────────────
        phys_scale = y.get('phys_scale', None)
        if phys_scale is None:
            return out_text                       # scale=0 → no change

        if isinstance(phys_scale, torch.Tensor):
            phys_scale = phys_scale.view(-1, 1, 1, 1)

        return out_text + phys_scale * (out_phys - out_text)

    # ------------------------------------------------------------------
    def __getattr__(self, name, default=None):
        return wrapped_getattr(self, name, default=None)
