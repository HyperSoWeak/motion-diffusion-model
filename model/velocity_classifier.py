"""
Simple velocity classifier for classifier-guided diffusion.
Predicts root velocity (xz plane linear velocity) from motion representations.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class VelocityRegressor(nn.Module):
    """
    Velocity regressor that predicts root velocity from MDM motion features.
    
    Takes MDM model output and predicts root linear velocity (vx, vz) for each frame.
    The root velocity information is extracted from channels [1:3] of the motion features,
    which contain the root linear velocity in the xz plane during training.
    
    Args:
        njoints: Number of joints
        nfeats: Number of features per joint
        latent_dim: Dimension of latent representation for the FC layers
        num_layers: Number of hidden layers
        dropout: Dropout rate
    """
    
    def __init__(self, njoints=22, nfeats=12, latent_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = njoints * nfeats  # Total flattened features
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Temporal processing: Conv1d to process motion features across time
        # Input: (B, njoints*nfeats, T) after reshaping
        # Process full motion to predict velocity at each frame
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(self.input_feats, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Hidden layers processing temporal features
        hidden_layers = []
        for i in range(num_layers - 1):
            hidden_layers.append(nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1))
            hidden_layers.append(nn.BatchNorm1d(latent_dim))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(dropout))
        
        self.hidden = nn.Sequential(*hidden_layers)
        
        # Output layer: predict velocity at each frame (2D: vx, vz)
        self.velocity_head = nn.Conv1d(latent_dim, 2, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, njoints, nfeats, seq_len) motion features from MDM
        
        Returns:
            vel: (batch_size, seq_len, 2) root velocity [vel_x, vel_z] at each frame
        """
        B, njoints, nfeats, T = x.shape
        
        # Reshape to (batch_size, njoints*nfeats, seq_len) for processing
        x_flat = x.reshape(B, njoints * nfeats, T)
        
        # Process through temporal encoder
        h = self.temporal_encoder(x_flat)  # (B, latent_dim, T)
        
        # Process through hidden layers
        h = self.hidden(h)  # (B, latent_dim, T)
        
        # Predict velocity at each frame
        vel = self.velocity_head(h)  # (B, 2, T)
        
        # Reshape to (batch_size, seq_len, 2)
        vel = vel.permute(0, 2, 1)  # (B, T, 2)
        
        return vel
    
    def forward_with_features(self, x):
        """
        Forward pass that also returns intermediate features for visualization.
        
        Args:
            x: (batch_size, njoints, nfeats, seq_len) motion features
        
        Returns:
            vel: (batch_size, seq_len, 2) predicted root velocity
            features: dict with intermediate activations
        """
        B, njoints, nfeats, T = x.shape
        x_flat = x.reshape(B, njoints * nfeats, T)
        
        h = self.temporal_encoder(x_flat)  # (B, latent_dim, T)
        h = self.hidden(h)  # (B, latent_dim, T)
        vel = self.velocity_head(h)  # (B, 2, T)
        
        vel = vel.permute(0, 2, 1)  # (B, T, 2)
        
        return vel, {'features': h}


class VelocityClassifier(nn.Module):
    """
    Classifier for velocity magnitude (slow, medium, fast motions).
    Can be used for categorical velocity control.
    """
    
    def __init__(self, input_feats=263, latent_dim=256, num_classes=3, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Input projection
        self.input_proj = nn.Linear(input_feats, latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim)
        
        # Hidden layers
        hidden_layers = []
        for i in range(num_layers - 1):
            hidden_layers.append(nn.Linear(latent_dim, latent_dim))
            hidden_layers.append(nn.LayerNorm(latent_dim))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(dropout))
        
        self.hidden = nn.Sequential(*hidden_layers)
        
        # Classification head
        self.classifier = nn.Linear(latent_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, njoints, nfeats, seq_len) motion features
        
        Returns:
            logits: (batch_size, num_classes) classification logits
        """
        B, C, _, T = x.shape
        
        x_pool = x.squeeze(2)  # (B, C, T)
        x_pool = self.pool(x_pool)  # (B, C, 1)
        x_pool = x_pool.squeeze(-1)  # (B, C)
        
        h = self.input_proj(x_pool)
        h = self.input_norm(h)
        h = F.relu(h)
        
        h = self.hidden(h)
        logits = self.classifier(h)  # (B, num_classes)
        
        return logits


def get_velocity_from_motion(motion_data):
    """
    Extract root velocity from motion data.
    
    Args:
        motion_data: (batch_size, njoints, nfeats, seq_len) 
                    or denormalized motion in original format
    
    Returns:
        velocities: (batch_size, 2) or (batch_size, seq_len, 2) average or per-frame velocities
    """
    if motion_data.dim() == 4:
        # Extract velocity channels [1:3] and average over time
        B, C, _, T = motion_data.shape
        # Root linear velocity is in channels [1:3] (xz plane)
        vel = motion_data[:, 1:3, 0, :]  # (B, 2, T)
        vel_avg = vel.mean(dim=-1)  # (B, 2)
        return vel_avg
    else:
        raise ValueError(f"Unexpected motion_data shape: {motion_data.shape}")


def compute_velocity_magnitude(velocity):
    """
    Compute L2 norm of velocity vector.
    
    Args:
        velocity: (..., 2) velocity vectors in xz plane
    
    Returns:
        magnitude: (...,) velocity magnitudes
    """
    return torch.norm(velocity, p=2, dim=-1)
