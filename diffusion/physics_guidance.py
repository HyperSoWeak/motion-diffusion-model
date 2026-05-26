"""
Physics-based guidance for MDM inference on HumanML3D (hml_vec, 263-dim).

Plug PhysicsGuidanceFn as `cond_fn` in p_sample_loop / ddim_sample_loop
with `cond_fn_with_grad=True`. It steers the denoising trajectory toward
physically plausible motions by penalising three artefacts:

  * Floor penetration  — feet passing below y = 0
  * Floating           — feet claiming ground-contact while floating above floor
  * Foot skating       — feet sliding while ground-contact label = 1

The gradient is computed w.r.t. the noisy x_t so that it acts exactly like
classifier guidance (Song et al. 2021) on the physics log-probability.

HumanML3D 263-dim feature layout (per frame):
  [0]     root rotation velocity
  [1:3]   root linear velocity XZ
  [3]     root Y (absolute pelvis height above floor)
  [4:67]  ric_data — joint positions relative to root (21 joints × 3)
  [67:193] rot_data — 6-D rotations (21 joints × 6)
  [193:259] local_velocity (22 joints × 3)
  [259:263] foot contact labels [L_ankle, L_foot, R_ankle, R_foot]

Foot joints in ric_data (ric-index = SMPL-joint-index − 1):
  L_Ankle → SMPL 7  → ric 6  → feat Y = 4 + 6*3 + 1  = 23
  L_Foot  → SMPL 10 → ric 9  → feat Y = 4 + 9*3 + 1  = 32
  R_Ankle → SMPL 8  → ric 7  → feat Y = 4 + 7*3 + 1  = 26
  R_Foot  → SMPL 11 → ric 10 → feat Y = 4 + 10*3 + 1 = 35

The ordering above matches the 4 foot-contact labels at [259:263].
"""

import torch
import torch.nn.functional as F


class PhysicsGuidanceFn:
    """
    Gradient-based physics guidance used as cond_fn in GaussianDiffusion.

    Usage
    -----
    ::
        guide = PhysicsGuidanceFn(mean, std,
                                   floor_weight=10., skate_weight=5.,
                                   float_weight=10.)
        sample = diffusion.p_sample_loop(
            model, shape,
            cond_fn=guide,
            cond_fn_with_grad=True,
            ...)

    Parameters
    ----------
    dataset_mean : Tensor [263]
        Per-feature mean used for denormalisation.
    dataset_std : Tensor [263]
        Per-feature std used for denormalisation.
    floor_weight : float
        Weight for the floor non-penetration penalty.
    skate_weight : float
        Weight for the foot-skating penalty.
    float_weight : float
        Weight for the floating (foot-near-floor-when-contact) penalty.
    guidance_scale : float
        Global scale applied to the returned gradient (like CFG scale).
    """

    # ric_data Y indices for each foot joint (order matches FC labels)
    # FC label order: [L_ankle, L_foot, R_ankle, R_foot]
    FOOT_RIC_Y_IDXS = [23, 32, 26, 35]
    ROOT_Y_IDX = 3
    FC_START = 259
    FC_END   = 263

    def __init__(self, dataset_mean, dataset_std,
                 floor_weight: float = 10.,
                 skate_weight: float = 5.,
                 float_weight: float = 10.,
                 guidance_scale: float = 1.):
        # Store as 1-D CPU tensors; move to device on first call.
        self._mean = dataset_mean.float().view(-1) if torch.is_tensor(dataset_mean) \
                     else torch.tensor(dataset_mean, dtype=torch.float32)
        self._std  = dataset_std.float().view(-1)  if torch.is_tensor(dataset_std) \
                     else torch.tensor(dataset_std,  dtype=torch.float32)
        self.floor_weight   = floor_weight
        self.skate_weight   = skate_weight
        self.float_weight   = float_weight
        self.guidance_scale = guidance_scale

    # ------------------------------------------------------------------
    def _denorm(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm : [bs, 263, 1, nframes]  normalised hml_vec
        returns : same shape, denormalised to physical units
        """
        mean = self._mean.to(x_norm.device).view(1, -1, 1, 1)
        std  = self._std .to(x_norm.device).view(1, -1, 1, 1)
        return x_norm * std + mean

    # ------------------------------------------------------------------
    def compute_energy(self, pred_x0: torch.Tensor) -> torch.Tensor:
        """
        Compute total physics energy from the predicted clean motion.

        pred_x0 : [bs, 263, 1, nframes]  normalised
        returns  : scalar energy  (higher ⟹ more violation)
        """
        x = self._denorm(pred_x0)               # [bs, 263, 1, nframes]

        root_y = x[:, self.ROOT_Y_IDX, 0, :]   # [bs, nframes]

        # Absolute foot heights: root_y + relative ric height
        foot_abs_y = torch.stack(
            [root_y + x[:, idx, 0, :] for idx in self.FOOT_RIC_Y_IDXS],
            dim=1,
        )   # [bs, 4, nframes]

        # Soft ground-contact labels (0 = air, 1 = ground)
        fc_raw  = x[:, self.FC_START:self.FC_END, 0, :]    # [bs, 4, nframes]
        fc_soft = torch.sigmoid(fc_raw * 5.)                # smooth threshold

        # 1. Floor non-penetration — penalise feet going below y = 0
        floor_energy = F.relu(-foot_abs_y).pow(2).sum()

        # 2. Foot skating — penalise horizontal/vertical velocity when grounded
        foot_vel   = (foot_abs_y[:, :, 1:] - foot_abs_y[:, :, :-1]).pow(2)
        fc_contact = fc_soft[:, :, :-1]                     # align with velocity
        skate_energy = (foot_vel * fc_contact).sum()

        # 3. Floating — when contact, foot should be near the floor (y ≈ 0)
        float_energy = (foot_abs_y.pow(2) * fc_soft).sum()

        energy = (self.floor_weight * floor_energy
                  + self.skate_weight * skate_energy
                  + self.float_weight * float_energy)
        return energy

    # ------------------------------------------------------------------
    def __call__(self, x, t, p_mean_var, **model_kwargs):
        """
        Gradient function compatible with GaussianDiffusion.condition_mean_with_grad
        and GaussianDiffusion.condition_score_with_grad.

        Returns -∇_x E(x̂₀(x))  which acts as ∇_x log p_physics(x).

        x         : [bs, 263, 1, nframes]  noisy x_t (requires_grad=True)
        t         : timestep tensor
        p_mean_var: dict with 'pred_xstart' key (connected to x via autograd)
        """
        pred_x0 = p_mean_var['pred_xstart']     # depends on x ⟹ gradient flows
        energy  = self.compute_energy(pred_x0)
        grad    = torch.autograd.grad(energy, x, retain_graph=False)[0]

        # Replace any NaN/Inf (can happen at early noisy timesteps) with 0,
        # then clamp each element to [-clip, +clip] to prevent explosion.
        # We preserve the relative gradient structure (unlike full normalisation)
        # so guidance_scale has a stable, interpretable meaning.
        grad = torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
        clip = grad.abs().mean().clamp(min=1e-8) * 100.   # clip at 100× mean-abs
        grad = grad.clamp(-clip, clip)

        return -grad * self.guidance_scale  # negate: gradient ascent on log p


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_physics_guidance(dataset, guidance_scale: float = 1.,
                           floor_weight: float = 10.,
                           skate_weight: float = 5.,
                           float_weight: float = 10.) -> PhysicsGuidanceFn:
    """
    Build a PhysicsGuidanceFn from a HumanML3D dataset object.

    Looks for:
      - dataset.mean_gpu / dataset.std_gpu  (GPU tensors, used during training)
      - dataset.t2m_dataset.mean / .std     (numpy arrays from t2m loader)
      - 'dataset/t2m_mean.npy' fallback

    Returns None if the dataset is not HumanML (non-hml_vec).
    """
    if getattr(dataset, 'dataname', None) not in (None,) and \
       not hasattr(dataset, 't2m_dataset'):
        return None  # non-HumanML dataset — skip

    mean, std = _get_mean_std(dataset)
    if mean is None:
        return None

    return PhysicsGuidanceFn(
        dataset_mean=torch.from_numpy(mean).float(),
        dataset_std=torch.from_numpy(std).float(),
        floor_weight=floor_weight,
        skate_weight=skate_weight,
        float_weight=float_weight,
        guidance_scale=guidance_scale,
    )


def _get_mean_std(dataset):
    import numpy as np
    import os

    # Prefer GPU tensors (training path)
    if hasattr(dataset, 'mean_gpu') and dataset.mean_gpu is not None:
        m = dataset.mean_gpu.cpu().numpy()
        s = dataset.std_gpu.cpu().numpy()
        return m.reshape(-1), s.reshape(-1)

    # t2m DataLoader wraps a t2m_dataset
    inner = getattr(dataset, 't2m_dataset', None)
    if inner is not None:
        if hasattr(inner, 'mean') and inner.mean is not None:
            return inner.mean.reshape(-1), inner.std.reshape(-1)

    # Fallback: load from disk
    for base in ['dataset', 'data_loaders']:
        m_path = os.path.join(base, 't2m_mean.npy')
        s_path = os.path.join(base, 't2m_std.npy')
        if os.path.exists(m_path) and os.path.exists(s_path):
            return np.load(m_path).reshape(-1), np.load(s_path).reshape(-1)

    print('[PhysicsGuidance] Warning: could not find dataset mean/std. Guidance disabled.')
    return None, None
