import torch


HML_FOOT_JOINTS = (7, 10, 8, 11)  # left_ankle, left_foot, right_ankle, right_foot


def joints_to_btj3(joints):
    """Return joints as [batch, frames, joints, xyz]."""
    if joints.dim() == 3:
        if joints.shape[-1] != 3:
            raise ValueError(f"Expected [frames, joints, 3], got {tuple(joints.shape)}")
        return joints.unsqueeze(0)

    if joints.dim() != 4:
        raise ValueError(f"Expected a 3D or 4D joints tensor, got {tuple(joints.shape)}")

    if joints.shape[-1] == 3:
        return joints
    if joints.shape[2] == 3:
        return joints.permute(0, 3, 1, 2)

    raise ValueError(
        "Could not infer joints layout. Expected [B,T,J,3] or [B,J,3,T], "
        f"got {tuple(joints.shape)}"
    )


def lengths_to_mask(lengths, num_frames, device=None):
    if lengths is None:
        return torch.ones(1, num_frames, dtype=torch.bool, device=device)
    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths, device=device)
    lengths = lengths.to(device=device)
    return torch.arange(num_frames, device=device).unsqueeze(0) < lengths.long().unsqueeze(1)


def masked_mean(values, mask=None, reduction="mean", eps=1e-8):
    if mask is None:
        per_sample = values.flatten(1).mean(dim=1)
    else:
        while mask.dim() < values.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(dtype=values.dtype, device=values.device)
        reduce_dims = tuple(range(1, values.dim()))
        per_sample = (values * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp_min(eps)

    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def ground_penetration_loss(joints, lengths=None, floor_height=0.0, reduction="mean"):
    joints = joints_to_btj3(joints)
    frame_mask = lengths_to_mask(lengths, joints.shape[1], joints.device)
    penetration = torch.relu(floor_height - joints[..., 1]).pow(2)
    return masked_mean(penetration, frame_mask, reduction=reduction)


def foot_skating_loss(
    joints,
    lengths=None,
    foot_indices=HML_FOOT_JOINTS,
    floor_height=0.0,
    contact_height_threshold=0.05,
    reduction="mean",
):
    joints = joints_to_btj3(joints)
    foot_pos = joints[:, :, list(foot_indices), :]
    foot_vel = foot_pos[:, 1:] - foot_pos[:, :-1]
    foot_height = foot_pos[:, :-1, :, 1]
    contact = foot_height <= (floor_height + contact_height_threshold)
    horizontal_speed_sq = foot_vel[..., [0, 2]].pow(2).sum(dim=-1)

    frame_mask = lengths_to_mask(lengths, joints.shape[1], joints.device)[:, :-1]
    contact_mask = contact & frame_mask.unsqueeze(-1)
    return masked_mean(horizontal_speed_sq, contact_mask, reduction=reduction)


def acceleration_smoothness_loss(joints, lengths=None, reduction="mean"):
    joints = joints_to_btj3(joints)
    vel = joints[:, 1:] - joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    acc_sq = acc.pow(2).sum(dim=-1)

    frame_mask = lengths_to_mask(lengths, joints.shape[1], joints.device)[:, 2:]
    return masked_mean(acc_sq, frame_mask, reduction=reduction)


def root_velocity_loss(joints, target_velocity, lengths=None, reduction="mean"):
    joints = joints_to_btj3(joints)
    root_vel = joints[:, 1:, 0, :] - joints[:, :-1, 0, :]
    if not torch.is_tensor(target_velocity):
        target_velocity = torch.as_tensor(target_velocity, dtype=joints.dtype, device=joints.device)
    target_velocity = target_velocity.to(dtype=joints.dtype, device=joints.device)
    while target_velocity.dim() < root_vel.dim():
        target_velocity = target_velocity.unsqueeze(0)
    error = (root_vel - target_velocity).pow(2).sum(dim=-1)

    frame_mask = lengths_to_mask(lengths, joints.shape[1], joints.device)[:, 1:]
    return masked_mean(error, frame_mask, reduction=reduction)


def anchor_loss(joints, reference_joints, lengths=None, reduction="mean"):
    joints = joints_to_btj3(joints)
    reference_joints = joints_to_btj3(reference_joints).to(dtype=joints.dtype, device=joints.device)
    error = (joints - reference_joints).pow(2).sum(dim=-1)
    frame_mask = lengths_to_mask(lengths, joints.shape[1], joints.device)
    return masked_mean(error, frame_mask, reduction=reduction)


def physics_loss(
    joints,
    lengths=None,
    lambda_ground=1.0,
    lambda_foot=1.0,
    lambda_smooth=0.0,
    lambda_root=0.0,
    target_velocity=None,
    floor_height=0.0,
    contact_height_threshold=0.05,
    foot_indices=HML_FOOT_JOINTS,
    reduction="mean",
):
    terms = {}
    if lambda_ground:
        terms["phys_ground"] = ground_penetration_loss(
            joints, lengths=lengths, floor_height=floor_height, reduction=reduction
        )
    if lambda_foot:
        terms["phys_foot"] = foot_skating_loss(
            joints,
            lengths=lengths,
            foot_indices=foot_indices,
            floor_height=floor_height,
            contact_height_threshold=contact_height_threshold,
            reduction=reduction,
        )
    if lambda_smooth:
        terms["phys_smooth"] = acceleration_smoothness_loss(joints, lengths=lengths, reduction=reduction)
    if lambda_root:
        if target_velocity is None:
            raise ValueError("target_velocity is required when lambda_root is non-zero")
        terms["phys_root"] = root_velocity_loss(
            joints, target_velocity=target_velocity, lengths=lengths, reduction=reduction
        )

    if reduction == "none":
        batch = joints_to_btj3(joints).shape[0]
        total = torch.zeros(batch, dtype=joints.dtype, device=joints.device)
    else:
        total = torch.zeros((), dtype=joints.dtype, device=joints.device)

    total = total + lambda_ground * terms.get("phys_ground", 0.0)
    total = total + lambda_foot * terms.get("phys_foot", 0.0)
    total = total + lambda_smooth * terms.get("phys_smooth", 0.0)
    total = total + lambda_root * terms.get("phys_root", 0.0)
    terms["phys_loss"] = total
    return total, terms


@torch.no_grad()
def physics_metrics(
    joints,
    lengths=None,
    foot_indices=HML_FOOT_JOINTS,
    floor_height=0.0,
    contact_height_threshold=0.05,
):
    joints = joints_to_btj3(joints)
    frame_mask = lengths_to_mask(lengths, joints.shape[1], joints.device)
    while frame_mask.shape[0] < joints.shape[0]:
        frame_mask = frame_mask.expand(joints.shape[0], -1)

    y = joints[..., 1]
    valid_joint_mask = frame_mask.unsqueeze(-1).expand_as(y)
    penetration = torch.relu(floor_height - y)
    penetrating = (penetration > 0) & valid_joint_mask

    foot_pos = joints[:, :, list(foot_indices), :]
    foot_vel = foot_pos[:, 1:] - foot_pos[:, :-1]
    contact = foot_pos[:, :-1, :, 1] <= (floor_height + contact_height_threshold)
    valid_contact = contact & frame_mask[:, :-1].unsqueeze(-1)
    horizontal_speed = torch.linalg.norm(foot_vel[..., [0, 2]], dim=-1)

    vel = joints[:, 1:] - joints[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    acc_norm = torch.linalg.norm(acc, dim=-1)
    acc_mask = frame_mask[:, 2:].unsqueeze(-1).expand_as(acc_norm)

    return {
        "ground_penetration_rate": _safe_ratio(penetrating.sum(), valid_joint_mask.sum()).item(),
        "mean_penetration_depth": masked_mean(penetration, valid_joint_mask, reduction="mean").item(),
        "max_penetration_depth": penetration[valid_joint_mask].max().item() if valid_joint_mask.any() else 0.0,
        "foot_contact_rate": _safe_ratio(valid_contact.sum(), frame_mask[:, :-1].sum() * len(foot_indices)).item(),
        "foot_skating_score": masked_mean(horizontal_speed, valid_contact, reduction="mean").item(),
        "mean_acceleration": masked_mean(acc_norm, acc_mask, reduction="mean").item(),
    }


def _safe_ratio(numerator, denominator):
    denominator = torch.as_tensor(denominator, dtype=torch.float32, device=numerator.device)
    return numerator.float() / denominator.clamp_min(1.0)
