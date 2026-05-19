"""
Training script for velocity regressor on HumanML3D dataset with noise scheduling.

The velocity regressor learns to predict root velocity (vx, vz) from noisy motion features
across the diffusion noise schedule. This follows proper classifier-guided diffusion methodology:
- Train on noisy motions (x_t) at various diffusion timesteps, not just clean data
- Regressor learns to extract velocity signal even from partially-denoised states
- This makes guidance effective during all diffusion sampling steps

IMPORTANT: This script trains the regressor to match the MDM model's motion representation.
For HumanML3D, the MDM model outputs motion as (batch, njoints, nfeats, seq_len).
The actual dimensions depend on the loaded model, but default to njoints=263, nfeats=1
(matching the 263 total motion features from HumanML3D dataset).

Usage:
    python train/train_velocity_regressor.py \
        --data_root dataset/HumanML3D \
        --output_dir checkpoints/velocity_regressor \
        --njoints 263 \
        --nfeats 1 \
        --epochs 50 \
        --batch_size 32 \
        --learning_rate 1e-3 \
        --num_diffusion_steps 1000 \
        --noise_schedule linear
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
from diffusion.gaussian_diffusion import get_named_beta_schedule


def get_noise_schedule(num_steps, schedule_type='linear'):
    """
    Get beta schedule for diffusion process.
    
    Args:
        num_steps: number of diffusion steps
        schedule_type: 'linear' or 'cosine'
    
    Returns:
        betas: (num_steps,) noise schedule
        alphas_cumprod: (num_steps,) cumulative product of (1-beta)
    """
    betas = get_named_beta_schedule(schedule_type, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return betas, alphas_cumprod


def apply_noise_to_motion(motion_clean, t, alphas_cumprod):
    """
    Add noise to clean motion at diffusion step t.
    
    Args:
        motion_clean: (seq_len, nfeats) clean motion
        t: diffusion timestep (0 to num_steps-1)
        alphas_cumprod: cumulative product schedule
    
    Returns:
        motion_noisy: (seq_len, nfeats) noisy motion
        noise: (seq_len, nfeats) the noise that was added
    """
    sqrt_alpha_cumprod = np.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - alphas_cumprod[t])
    
    # Generate random Gaussian noise
    noise = np.random.randn(*motion_clean.shape)
    
    # x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
    motion_noisy = sqrt_alpha_cumprod * motion_clean + sqrt_one_minus_alpha_cumprod * noise
    
    return motion_noisy.astype(np.float32), noise.astype(np.float32), t


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
    batch_timesteps = []
    
    for item in batch:
        motion = item['motion']  # (seq_len, nfeats)
        velocity = item['velocity']  # (seq_len, 2)
        length = item['length']
        timestep = item.get('timestep', 0)  # diffusion timestep
        
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
        batch_timesteps.append(timestep)
    
    return {
        'motion': torch.stack(batch_motion, dim=0),  # (B, max_length, nfeats)
        'velocity': torch.stack(batch_velocity, dim=0),  # (B, max_length, 2)
        'length': torch.tensor(batch_lengths),
        'timestep': torch.tensor(batch_timesteps)  # (B,) diffusion timesteps
    }


class VelocityDataset(Dataset):
    """
    Dataset wrapper for velocity regression training with noise scheduling.
    
    Trains on noisy motions across diffusion timesteps (proper classifier guidance):
    - Each sample is a (motion_clean, timestep) pair
    - At runtime, noise is added to create (motion_noisy, target_velocity) training examples
    - Regressor learns to extract velocity from partially-denoised states
    
    Root velocity is stored in channels [1:3] of the motion feature representation.
    """
    
    def __init__(self, humanml_dataset, num_diffusion_steps=1000, noise_schedule='linear'):
        """
        Args:
            humanml_dataset: HumanML3D dataset instance
            num_diffusion_steps: number of diffusion timesteps
            noise_schedule: 'linear' or 'cosine'
        """
        self.dataset = humanml_dataset
        self.motion_data = []
        self.lengths = []
        self.num_diffusion_steps = num_diffusion_steps
        
        # Get noise schedule
        _, self.alphas_cumprod = get_noise_schedule(num_diffusion_steps, noise_schedule)
        
        print("Processing HumanML3D dataset for velocity regression...")
        for idx in tqdm(range(len(self.dataset)), desc="Processing motions"):
            item = self.dataset[idx]
            motion = item[-3]  # motion array from dataset
            length = item[-2]   # sequence length
            
            # Motion shape from dataset is (seq_len, nfeats)
            self.motion_data.append(motion)
            self.lengths.append(length)
        
    def __len__(self):
        # Return data size multiplied by number of noise levels
        # This way each motion is trained at multiple noise levels
        return len(self.motion_data)
    
    def __getitem__(self, idx):
        """
        Returns noisy motion and ground-truth velocity for a random diffusion timestep.
        
        Returns:
            motion: (seq_len, nfeats) NOISY motion at diffusion step t
            velocity: (seq_len, 2) ground truth velocity [vx, vz] from channels [1:3]
            length: sequence length
            timestep: diffusion timestep t (0 to num_steps-1)
        """
        motion = self.motion_data[idx]  # (seq_len, nfeats)
        length = self.lengths[idx]
        
        # Extract ground truth velocity from channels [1:3] of CLEAN motion
        # These channels contain root_linear_velocity in xz plane
        velocity = motion[:length, 1:3]  # (seq_len, 2)
        
        # Trim motion to actual length (remove padding)
        motion_clean = motion[:length, :]  # (seq_len, nfeats)
        
        # Sample random diffusion timestep for this motion
        # This ensures regressor sees velocity extraction task across all noise levels
        timestep = np.random.randint(0, self.num_diffusion_steps)
        
        # Add noise to motion
        motion_noisy, _, _ = apply_noise_to_motion(
            motion_clean, timestep, self.alphas_cumprod
        )
        
        return {
            'motion': torch.FloatTensor(motion_noisy),  # Noisy motion (x_t)
            'velocity': torch.FloatTensor(velocity),    # Clean velocity (from x_0)
            'length': length,
            'timestep': timestep
        }


class VelocityRegressorTrainer:
    """Training loop for velocity regressor.
    
    Ensures the regressor is trained on the exact same feature representation
    as the MDM model (with proper padding if needed).
    """
    
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
        """Train for one epoch on noisy motions."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            motion = batch['motion'].to(self.device)  # (B, seq_len, nfeats) - NOISY motion (x_t)
            velocity = batch['velocity'].to(self.device)  # (B, seq_len, 2) - clean velocity from x_0
            timesteps = batch['timestep'].to(self.device)  # (B,) diffusion timesteps
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
            
            # Predict velocity from NOISY motion at diffusion step t
            # This trains the regressor to extract velocity even from partially-denoised states
            pred_velocity = self.model(motion_mdm, timesteps)  # (B, seq_len, 2)
            
            # Loss: compare per-frame predictions with CLEAN ground truth velocity
            # The key insight: we train on noisy input but supervise with clean velocity
            # This forces the regressor to learn the underlying velocity despite noise
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
        """Validate on validation set (noisy motions)."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            motion = batch['motion'].to(self.device)  # (B, seq_len, nfeats) - NOISY motion
            velocity = batch['velocity'].to(self.device)  # (B, seq_len, 2) - clean velocity
            timesteps = batch['timestep'].to(self.device)  # (B,) diffusion timesteps
            
            B, T, nfeats = motion.shape
            
            # Reshape for MDM format: (B, seq_len, nfeats) -> (B, njoints, nfeats_per_joint, seq_len)
            # Pad to match model's expected input size
            nfeats_expected = self.njoints * self.nfeats_per_joint
            
            if nfeats < nfeats_expected:
                padding = torch.zeros(B, T, nfeats_expected - nfeats, device=motion.device)
                motion = torch.cat([motion, padding], dim=-1)
            
            motion_mdm = motion.permute(0, 2, 1)  # (B, nfeats, seq_len)
            motion_mdm = motion_mdm.view(B, self.njoints, self.nfeats_per_joint, T)  # (B, njoints, nfeats_per_joint, seq_len)
            
            # Predict from noisy motion
            pred_velocity = self.model(motion_mdm, timesteps)  # (B, seq_len, 2)
            
            # Loss: noisy input vs clean velocity target
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
        description='Train velocity regressor on HumanML3D dataset with noise scheduling'
    )
    
    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to HumanML3D dataset root')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split: train, val, or test')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of training data to use for validation')
    
    # Diffusion scheduling
    parser.add_argument('--num_diffusion_steps', type=int, default=1000,
                        help='Number of diffusion timesteps (matches MDM training)')
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Noise schedule type (should match MDM model)')
    
    # Model
    parser.add_argument('--njoints', type=int, default=263,
                        help='Number of joints (default 263 for HumanML3D - matches dataset total features)')
    parser.add_argument('--nfeats', type=int, default=1,
                        help='Features per joint (default 1 for HumanML3D - total input features = njoints * nfeats)')
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
    
    # Create velocity dataset with noise scheduling
    vel_dataset = VelocityDataset(
        dataset, 
        num_diffusion_steps=args.num_diffusion_steps,
        noise_schedule=args.noise_schedule
    )
    
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
    # Use explicit njoints and nfeats from command line
    # These MUST match the loaded MDM model's dimensions to avoid shape mismatches
    # Default: njoints=263, nfeats=1 (matches HumanML3D dataset: 263 total features)
    njoints = args.njoints
    nfeats_per_joint = args.nfeats
    total_features = njoints * nfeats_per_joint
    dataset_features = dataset.mean.shape[0]
    
    print(f"Dataset motion features: {dataset_features}")
    print(f"Training regressor with: njoints={njoints}, nfeats={nfeats_per_joint}")
    print(f"Regressor input features: {total_features}")
    if dataset_features != total_features:
        print(f"  → Dataset will be padded/trimmed from {dataset_features} to {total_features} features")
    print("\nMake sure these match your MDM model's (njoints, nfeats) when loading!")
    
    model = VelocityRegressor(
        njoints=njoints,
        nfeats=nfeats_per_joint,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_timestep_cond=True,  # Enable timestep conditioning
        num_diffusion_steps=args.num_diffusion_steps
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
    print(f"  python -m sample.generate \\")
    print(f"    --model_path path/to/your/mdm_model.pt \\")
    print(f"    --velocity_guidance_scale 1.0 \\")
    print(f"    --velocity_regressor_path {args.output_dir}/velocity_regressor_best.pt \\")
    print(f"    --text_prompt 'your prompt' \\")
    print(f"    --target_velocity '1.0,0.0'")


if __name__ == '__main__':
    main()
