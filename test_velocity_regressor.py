#!/usr/bin/env python
"""Quick test of velocity regressor model"""

import torch
from model.velocity_classifier import VelocityRegressor

# Test model initialization with correct parameters
print("Testing VelocityRegressor with njoints=22, nfeats=12...")

model = VelocityRegressor(
    njoints=22,
    nfeats=12,
    latent_dim=256,
    num_layers=3,
    dropout=0.1
)

print(f"✓ Model created successfully")
print(f"  Input features: {model.input_feats}")
print(f"  Latent dim: {model.latent_dim}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
B, njoints, nfeats, T = 2, 22, 12, 100
x = torch.randn(B, njoints, nfeats, T)

with torch.no_grad():
    output = model(x)

print(f"\n✓ Forward pass successful")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Expected output shape: ({B}, {T}, 2)")

assert output.shape == (B, T, 2), f"Output shape mismatch: {output.shape} vs ({B}, {T}, 2)"

print("\n✅ All tests passed!")
