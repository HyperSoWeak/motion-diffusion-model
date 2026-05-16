"""
Training script for velocity regressor on HumanML3D dataset.

The velocity regressor learns to predict root velocity (vx, vz) from motion features.
This model is used for classifier-guided diffusion to control motion trajectories.

Usage:
    python train/train_velocity_regressor.py \
        --data_root dataset/HumanML3D \
        --output_dir checkpoints/velocity_regressor \
        --epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-3
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.velocity_classifier import VelocityRegressor
from data_loaders.humanml.data.dataset import HumanML3D
from utils.fixseed import fixseed


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads all sequences to the maximum length in the batch.
    """
    # Find max length in batch
    max_length = max(b['length'] for b in batch)
    
    # Pad motion and velocity to max_length
    batch_motion = []
    batch_velocity = []
    batch_lengths = []
    
    for item in batch:
        motion = item['motion']  # (seq_len, nfeats)
        velocity = item['velocity']  # (seq_len, 2)
        length = item['length']
        
        # Pad motion
        if len(motion) < max_length:
            padding = torch.zeros(max_length - len(motion), motion.shape[1])
            motion_padded = torch.cat([motion, padding], dim=0)
        else:
            motion_padded = motion
            
        # Pad velocity
        if len(velocity) < max_length:
            padding = torch.zeros(max_length - len(velocity), 2)
            velocity_padded = torch.cat([velocity, padding], dim=0)
        else:
            velocity_padded = velocity
        
        batch_motion.append(motion_padded)
        batch_velocity.append(velocity_padded)
        batch_lengths.append(length)
    
    return {
        'motion': torch.stack(batch_motion, dim=0),  # (B, max_length, nfeats)
        'velocity': torch.stack(batch_velocity, dim=0),  # (B, max_length, 2)
        'length': torch.tensor(batch_lengths)
    }


class VelocityDataset(Dataset):
    """
    Dataset wrapper for velocity regression training.
    Extracts motion sequences in MDM format and their corresponding root velocities.
    
    Root velocity is stored in channels [1:3] of the motion feature representation:
    - Channel 0: root_rot_velocity
    - Channels 1-2: root_linear_velocity (vx, vz) in xz plane
    - Channel 3: root_y
    - Rest: other motion features
    """
    
    def __init__(self, humanml_dataset):
        """
        Args:
            humanml_dataset: HumanML3D dataset instance
        """
        self.dataset = humanml_dataset
        self.motion_data = []
        self.lengths = []
        
        print("Processing HumanML3D dataset for velocity regression...")
        for idx in tqdm(range(len(self.dataset)), desc="Processing motions"):
            item = self.dataset[idx]
            motion = item[-3]  # motion array from dataset
            length = item[-2]   # sequence length
            
            # Motion shape from dataset is (seq_len, nfeats)
            # nfeats = joints_num * features_per_joint
            # Keep as is, we'll reshape later during __getitem__
            self.motion_data.append(motion)
            self.lengths.append(length)
        
    def __len__(self):
        return len(self.motion_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            motion: (seq_len, nfeats) motion features
            velocity: (seq_len, 2) ground truth velocity [vx, vz] from channels [1:3]
            length: sequence length
        """
        motion = self.motion_data[idx]  # (seq_len, nfeats)
        length = self.lengths[idx]
        
        # Extract ground truth velocity from channels [1:3]
        # These channels contain root_linear_velocity in xz plane
        velocity = motion[:length, 1:3]  # (seq_len, 2)
        
        # Trim motion to actual length (remove padding)
        motion_trimmed = motion[:length, :]  # (seq_len, nfeats)
        
        return {
            'motion': torch.FloatTensor(motion_trimmed),
            'velocity': torch.FloatTensor(velocity),
            'length': length
        }


class VelocityRegressorTrainer:
    """Training loop for velocity regressor."""
    
    def __init__(self, model, device, args, train_loader=None):
        """
        Args:
            model: VelocityRegressor instance
            device: torch device
            args: training arguments
            train_loader: DataLoader for accessing dataset info
        """
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.train_loader = train_loader
        
        # Extract dataset info for reshaping
        if train_loader is not None and hasattr(train_loader.dataset, 'dataset'):
            self.njoints = getattr(train_loader.dataset.dataset, 'njoints', 22)
        else:
            self.njoints = 22  # Default fallback
        
        # Will be set from main
        self.nfeats_per_joint = None
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-5
        )
        self.loss_fn = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # Checkpoints directory
        self.ckpt_dir = Path(args.output_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            motion = batch['motion'].to(self.device)  # (B, seq_len, nfeats)
            velocity = batch['velocity'].to(self.device)  # (B, seq_len, 2)
            lengths = batch['length']
            
            B, T, nfeats = motion.shape
            
            # Reshape for MDM format: (B, seq_len, nfeats) -> (B, njoints, nfeats_per_joint, seq_len)
            # Pad to match model's expected input size
            nfeats_expected = self.njoints * self.nfeats_per_joint
            
            if nfeats < nfeats_expected:
                padding = torch.zeros(B, T, nfeats_expected - nfeats, device=motion.device)
                motion = torch.cat([motion, padding], dim=-1)
            
            motion_mdm = motion.permute(0, 2, 1)  # (B, nfeats, seq_len)
            motion_mdm = motion_mdm.view(B, self.njoints, self.nfeats_per_joint, T)  # (B, njoints, nfeats_per_joint, seq_len)
            
            # Predict velocity for each frame
            pred_velocity = self.model(motion_mdm)  # (B, seq_len, 2)
            
            # Loss: compare per-frame predictions with ground truth
            # Both have shape (B, seq_len, 2)
            loss = self.loss_fn(pred_velocity, velocity)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics: mean absolute error per frame
            mae = torch.abs(pred_velocity - velocity).mean().item()
            
            total_loss += loss.item() * B
            total_mae += mae * B
            num_samples += B
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'mae': f'{mae:.6f}'})
        
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        
        return avg_loss, avg_mae
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            motion = batch['motion'].to(self.device)  # (B, seq_len, nfeats)
            velocity = batch['velocity'].to(self.device)  # (B, seq_len, 2)
            
            B, T, nfeats = motion.shape
            
            # Reshape for MDM format: (B, seq_len, nfeats) -> (B, njoints, nfeats_per_joint, seq_len)
            # Pad to match model's expected input size
            nfeats_expected = self.njoints * self.nfeats_per_joint
            
            if nfeats < nfeats_expected:
                padding = torch.zeros(B, T, nfeats_expected - nfeats, device=motion.device)
                motion = torch.cat([motion, padding], dim=-1)
            
            motion_mdm = motion.permute(0, 2, 1)  # (B, nfeats, seq_len)
            motion_mdm = motion_mdm.view(B, self.njoints, self.nfeats_per_joint, T)  # (B, njoints, nfeats_per_joint, seq_len)
            
            # Predict
            pred_velocity = self.model(motion_mdm)  # (B, seq_len, 2)
            
            # Loss
            loss = self.loss_fn(pred_velocity, velocity)
            mae = torch.abs(pred_velocity - velocity).mean().item()
            
            total_loss += loss.item() * B
            total_mae += mae * B
            num_samples += B
        
        avg_loss = total_loss / num_samples
        avg_mae = total_mae / num_samples
        
        return avg_loss, avg_mae
    
    def train(self, train_loader, val_loader=None):
        """Full training loop."""
        best_val_loss = float('inf')
        best_epoch = 0
        
        print(f"Starting training for {self.args.epochs} epochs...")
        print(f"Training on {len(train_loader.dataset)} samples")
        if val_loader:
            print(f"Validating on {len(val_loader.dataset)} samples")
        print()
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            
            # Validate
            if val_loader:
                val_loss, val_mae = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)
                
                print(f"Train Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
                print(f"Val Loss:   {val_loss:.6f}, MAE: {val_mae:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    self.save_checkpoint(f'velocity_regressor_best.pt')
            else:
                print(f"Train Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
                self.save_checkpoint(f'velocity_regressor_epoch{epoch:03d}.pt')
            
            # Learning rate scheduling
            self.scheduler.step()
        
        # Save final model
        self.save_checkpoint('velocity_regressor_final.pt')
        
        # Save training history
        self.save_history()
        
        print(f"\nTraining completed!")
        if val_loader:
            print(f"Best model saved at epoch {best_epoch}")
        
        return best_val_loss
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        ckpt_path = self.ckpt_dir / filename
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")
    
    def save_history(self):
        """Save training history."""
        history_path = self.ckpt_dir / 'training_history.json'
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, values in self.history.items():
            history_json[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        print(f"Saved: {history_path}")


def create_argument_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Train velocity regressor on HumanML3D dataset'
    )
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to HumanML3D dataset root')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split: train, val, or test')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension for regressor')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/velocity_regressor',
                        help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser


def main():
    """Main training function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set seed
    fixseed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading HumanML3D dataset from {args.data_root}")
    dataset = HumanML3D(
        mode=args.split,
        data_root=args.data_root,
        split=args.split,
        load_humanml=False  # Don't need HumanML features for velocity
    )
    print(f"Loaded {len(dataset)} motion sequences")
    
    # Create velocity dataset
    vel_dataset = VelocityDataset(dataset)
    
    # Split into train/val
    num_val = int(len(vel_dataset) * args.validation_split)
    num_train = len(vel_dataset) - num_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        vel_dataset,
        [num_train, num_val]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    # Infer motion dimensions from the mean/std
    # Total features = 4 + (joints_num-1)*9 + joints_num*3 + 4
    # which simplifies to: joints_num*12 - 1 = nfeats
    # So: joints_num = (nfeats + 1) / 12
    total_features = dataset.mean.shape[0]
    njoints = (total_features + 1) // 12
    nfeats_per_joint = (total_features + njoints - 1) // njoints  # Round up to get nfeats_per_joint
    
    print(f"Motion features: {total_features}")
    print(f"Number of joints: {njoints}")
    print(f"Features per joint: {nfeats_per_joint}")
    
    model = VelocityRegressor(
        njoints=njoints,
        nfeats=nfeats_per_joint,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = VelocityRegressorTrainer(model, device, args, train_loader)
    trainer.njoints = njoints  # Set the correct njoints
    trainer.nfeats_per_joint = nfeats_per_joint
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training complete!")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("\nTo use the trained model for velocity guidance:")
    print(f"  python sample/generate.py \\")
    print(f"    --model_path path/to/your/mdm_model.pt \\")
    print(f"    --velocity_guidance_scale 0.3 \\")
    print(f"    --velocity_regressor_path {args.output_dir}/velocity_regressor_best.pt \\")
    print(f"    --text_prompt 'your prompt' \\")
    print(f"    --target_velocity '1.0,0.0'")


if __name__ == '__main__':
    main()
