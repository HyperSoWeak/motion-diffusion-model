#!/usr/bin/env python
"""Test the velocity regressor training pipeline"""

import torch
import sys
sys.path.insert(0, '.')

from model.velocity_classifier import VelocityRegressor
from train.train_velocity_regressor import VelocityDataset, collate_fn
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("Testing Velocity Regressor Training Pipeline")
print("=" * 60)

# 1. Test model creation
print("\n1. Creating model with njoints=22, nfeats=12...")
model = VelocityRegressor(njoints=22, nfeats=12, latent_dim=256, num_layers=3)
print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Create dummy batch
print("\n2. Creating dummy training batch...")
B, njoints, nfeats, T = 4, 22, 12, 100
motion_batch = torch.randn(B, njoints, nfeats, T)
velocity_batch = torch.randn(B, T, 2)

print(f"   Motion shape: {motion_batch.shape}")
print(f"   Velocity shape: {velocity_batch.shape}")

# 3. Test forward pass
print("\n3. Testing forward pass...")
with torch.no_grad():
    pred_velocity = model(motion_batch)
print(f"   ✓ Output shape: {pred_velocity.shape}")
assert pred_velocity.shape == (B, T, 2), "Output shape mismatch!"

# 4. Test training step
print("\n4. Testing training step...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

motion_batch = motion_batch.to(device)
velocity_batch = velocity_batch.to(device)

# Forward pass
pred_velocity = model(motion_batch)
loss = loss_fn(pred_velocity, velocity_batch)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"   ✓ Loss: {loss.item():.6f}")
print(f"   ✓ Gradient norm: {sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5:.6f}")

# 5. Test data reshaping (as done in training loop)
print("\n5. Testing data reshaping for MDM format...")
seq_len_flat = 263  # Original flat motion dimension
njoints, nfeats_per_joint = 22, 12
nfeats_expected = njoints * nfeats_per_joint  # 264

# Simulate padded motion
motion_flat = torch.randn(B, T, nfeats_expected)

# Reshape as done in trainer
motion_mdm = motion_flat.permute(0, 2, 1)  # (B, nfeats, T)
motion_mdm = motion_mdm.view(B, njoints, nfeats_per_joint, T)

print(f"   Input (flat): {motion_flat.shape}")
print(f"   Reshaped (MDM): {motion_mdm.shape}")
assert motion_mdm.shape == (B, njoints, nfeats_per_joint, T)

# Verify model can process it
with torch.no_grad():
    output = model(motion_mdm.to(device))
print(f"   ✓ Model output: {output.shape}")

print("\n" + "=" * 60)
print("✅ All pipeline tests passed!")
print("=" * 60)
