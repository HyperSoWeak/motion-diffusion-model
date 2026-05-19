"""
Velocity-guided diffusion sampling.
Uses gradient-based guidance to control root velocity during generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from model.velocity_classifier import VelocityRegressor, compute_velocity_magnitude
from utils.misc import wrapped_getattr


class VelocityGuidedSampleModel(nn.Module):
    """
    Wrapper for velocity-guided diffusion sampling.
    
    Uses classifier-free guidance combined with velocity prediction to guide 
    motion generation toward desired root velocity trajectories.
    
    The guidance works by:
    1. Computing gradients of predicted velocity w.r.t. motion at each step
    2. Using these gradients to steer the denoising process
    3. Optionally combining with classifier-free guidance for text conditioning
    """
    
    def __init__(self, model, velocity_regressor=None, use_classifier_free=True):
        """
        Args:
            model: The MDM model to wrap
            velocity_regressor: VelocityRegressor for predicting velocity from motion
            use_classifier_free: Whether to also apply classifier-free guidance
        """
        super().__init__()
        self.model = model
        self.use_classifier_free = use_classifier_free
        
        # Initialize velocity regressor if not provided
        if velocity_regressor is None:
            self.velocity_regressor = VelocityRegressor(
                njoints=model.njoints,
                nfeats=model.nfeats,
                latent_dim=256,
                num_layers=3,
                dropout=0.1,
                use_timestep_cond=True,
                num_diffusion_steps=1000
            )
        else:
            self.velocity_regressor = velocity_regressor
        
        # Pointers to inner model properties
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        
        # Check if model has encode_text for text conditioning
        if hasattr(self.model, 'encode_text'):
            self.encode_text = self.model.encode_text
        
        # Velocity guidance parameters (set by caller)
        self.velocity_guidance_scale = 0.0
        self.target_velocity = None
        self.velocity_guidance_mode = 'direction'
        self._guidance_step = 0  # For logging
    
    def forward(self, x, timesteps, y=None):
        """
        Forward pass with velocity guidance.
        
        Args:
            x: (B, C, 1, T) noisy motion at diffusion step
            timesteps: (B,) diffusion timesteps
            y: dict with conditioning information
        
        Returns:
            output: (B, C, 1, T) model predictions with velocity guidance applied
        """
        if y is None:
            y = {}
        
        # Get base model prediction
        if self.use_classifier_free and 'scale' in y:
            # Apply classifier-free guidance
            y_uncond = deepcopy(y)
            y_uncond['uncond'] = True
            out = self.model(x, timesteps, y)
            out_uncond = self.model(x, timesteps, y_uncond)
            base_out = out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
        else:
            base_out = self.model(x, timesteps, y)
        
        # Apply velocity guidance if scale > 0
        if self.velocity_guidance_scale > 0:
            # Enable gradient computation for velocity guidance
            x_pred = base_out.clone().detach().requires_grad_(True)
            
            # Temporarily enable gradient computation in regressor
            # (needed even though regressor is in eval mode, so batch norm works during backward)
            with torch.enable_grad():
                # Predict velocity from the model output
                # VelocityRegressor returns (B, seq_len, 2) - per-frame predictions
                # Pass timesteps for timestep conditioning in the regressor
                vel_pred = self.velocity_regressor(x_pred, timesteps)  # (B, seq_len, 2)
                
                # Average velocity prediction across sequence for guidance signal
                vel_avg = vel_pred.mean(dim=1)  # (B, 2)
                
                # Debug: log predicted velocity
                self._guidance_step = getattr(self, '_guidance_step', 0) + 1
                if self._guidance_step % 50 == 0:  # Log every 50 steps
                    print(f"[Velocity Guidance] Step {self._guidance_step}")
                    print(f"  Predicted velocity (avg): {vel_avg[0].detach().cpu().numpy()}")
                    if self.target_velocity is not None:
                        print(f"  Target velocity: {self.target_velocity[0].detach().cpu().numpy()}")

                
                # Compute guidance signal based on mode
                if self.target_velocity is not None:
                    # Guidance towards specific velocity
                    if self.velocity_guidance_mode == 'direction':
                        # Guide towards target velocity direction
                        # self.target_velocity: (B, 2), vel_avg: (B, 2)
                        target_dir = F.normalize(self.target_velocity, p=2, dim=-1)  # (B, 2)
                        pred_dir = F.normalize(vel_avg, p=2, dim=-1)  # (B, 2)
                        # Negative loss: we want to maximize dot product (minimize negative)
                        guidance_loss = -torch.sum(target_dir * pred_dir, dim=-1).mean()
                    elif self.velocity_guidance_mode == 'magnitude':
                        # Guide towards target velocity magnitude
                        target_mag = compute_velocity_magnitude(self.target_velocity)  # (B,)
                        pred_mag = compute_velocity_magnitude(vel_avg)  # (B,)
                        guidance_loss = F.mse_loss(pred_mag, target_mag)
                    else:
                        raise ValueError(f"Unknown guidance mode: {self.velocity_guidance_mode}")
                else:
                    # Default: encourage forward motion (positive velocity in x direction)
                    forward_dir = torch.tensor([1.0, 0.0], device=vel_avg.device, dtype=vel_avg.dtype)
                    vel_norm = F.normalize(vel_avg, p=2, dim=-1)  # (B, 2)
                    guidance_loss = -torch.sum(forward_dir * vel_norm, dim=-1).mean()
                
                # Compute gradients
                guidance_loss.backward()
            
                # Apply gradient to output
                if x_pred.grad is not None:
                    grad = x_pred.grad
                    grad_norm_before_norm = torch.norm(grad).item()
                    # Normalize gradient to avoid too aggressive updates
                    grad = grad / (torch.norm(grad, p=2, keepdim=True) + 1e-8)
                    
                    if self._guidance_step % 50 == 0:  # Log every 50 steps
                        print(f"  Loss: {guidance_loss.item():.6f}")
                        print(f"  Grad norm before normalization: {grad_norm_before_norm:.6f}")
                        print(f"  Scaled update magnitude: {(self.velocity_guidance_scale * torch.norm(grad)).item():.4f}")
                        print(f"  base_out norm: {torch.norm(base_out).item():.4f}")
                        print(f"{base_out[0, :5, 0, :5].detach().cpu().numpy()}")  # Print a small slice of the output for debugging
                        print(f"{grad[0, :5, 0, :5].detach().cpu().numpy()}")  # Print corresponding slice of the gradient
                    
                    base_out = base_out - self.velocity_guidance_scale * grad
        
        return base_out
    
    def __getattr__(self, name, default=None):
        """Delegate attribute access to inner model."""
        if name in ['model', 'velocity_regressor', 'rot2xyz', 'translation', 
                    'njoints', 'nfeats', 'data_rep', 'cond_mode', 'encode_text',
                    'use_classifier_free', 'velocity_guidance_scale', 'target_velocity',
                    'velocity_guidance_mode']:
            return super().__getattr__(name)
        return wrapped_getattr(self, name, default=None)


class SimpleVelocityGuidance:
    """
    Simpler velocity guidance function for use during diffusion sampling loops.
    Can be passed as cond_fn to p_sample_loop.
    """
    
    def __init__(self, velocity_regressor, target_velocity=None, 
                 guidance_scale=1.0, guidance_mode='direction'):
        """
        Args:
            velocity_regressor: VelocityRegressor model
            target_velocity: (B, 2) target velocity or None
            guidance_scale: guidance strength
            guidance_mode: 'direction' or 'magnitude'
        """
        self.velocity_regressor = velocity_regressor
        self.target_velocity = target_velocity
        self.guidance_scale = guidance_scale
        self.guidance_mode = guidance_mode
        self._step_count = 0
    
    def __call__(self, x, t):
        """
        Compute guidance gradient for motion x at timestep t.
        
        Args:
            x: (B, njoints, nfeats, T) motion features
            t: timestep (not used directly, but available if needed)
        
        Returns:
            grad: guidance gradient for motion descent
        """
        self._step_count += 1
        
        if self.guidance_scale <= 0:
            return torch.zeros_like(x)
        
        # Ensure x requires gradient
        x_input = x.clone().detach().requires_grad_(True)
        
        # Compute guidance with gradient tracking enabled
        with torch.enable_grad():
            # Predict velocity: returns (B, seq_len, 2)
            # Pass timestep t for timestep conditioning in the regressor
            # t is a scalar timestep index
            t_batch = torch.full((x_input.shape[0],), t, dtype=torch.long, device=x_input.device)
            vel_pred = self.velocity_regressor(x_input, t_batch)
            vel_avg = vel_pred.mean(dim=1)  # (B, 2) - average across time
            
            if self._step_count % 50 == 0:
                print(f"[SimpleVelocityGuidance] Step {self._step_count}")
                print(f"  Predicted velocity (avg): {vel_avg[0].detach().cpu().numpy()}")
                if self.target_velocity is not None:
                    print(f"  Target velocity: {self.target_velocity[0].detach().cpu().numpy()}")
            
            # Compute loss
            if self.target_velocity is not None:
                if self.guidance_mode == 'direction':
                    target_dir = F.normalize(self.target_velocity, p=2, dim=-1)  # (B, 2)
                    pred_dir = F.normalize(vel_avg, p=2, dim=-1)  # (B, 2)
                    # Negative: we want to maximize alignment
                    loss = -torch.sum(target_dir * pred_dir, dim=-1).mean()
                elif self.guidance_mode == 'magnitude':
                    target_mag = compute_velocity_magnitude(self.target_velocity)  # (B,)
                    pred_mag = compute_velocity_magnitude(vel_avg)  # (B,)
                    loss = F.mse_loss(pred_mag, target_mag)
                else:
                    raise ValueError(f"Unknown guidance mode: {self.guidance_mode}")
            else:
                # Encourage forward motion
                forward_dir = torch.tensor([1.0, 0.0], device=vel_avg.device, dtype=vel_avg.dtype)
                vel_norm = F.normalize(vel_avg, p=2, dim=-1)  # (B, 2)
                loss = -torch.sum(forward_dir * vel_norm, dim=-1).mean()
            
            # Compute and normalize gradient
            loss.backward()
        
        if x_input.grad is not None:
            grad = x_input.grad
            grad_norm_before = torch.norm(grad).item()
            grad = grad / (torch.norm(grad, p=2, keepdim=True) + 1e-8)
            
            if self._step_count % 50 == 0:
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Grad norm before normalization: {grad_norm_before:.6f}")
                print(f"  Scaled update magnitude: {(self.guidance_scale * torch.norm(grad)).item():.4f}")
            
            return -grad * self.guidance_scale  # Negative gradient for descent
        else:
            return torch.zeros_like(x_input)


def create_velocity_guidance_fn(velocity_regressor, target_velocity=None, 
                                guidance_scale=1.0, guidance_mode='direction'):
    """
    Factory function to create a velocity guidance function for use in sampling.
    
    Args:
        velocity_regressor: VelocityRegressor model
        target_velocity: (B, 2) target velocity
        guidance_scale: guidance strength
        guidance_mode: 'direction' or 'magnitude'
    
    Returns:
        cond_fn: function that can be used as cond_fn in p_sample_loop
    """
    guidance = SimpleVelocityGuidance(
        velocity_regressor=velocity_regressor,
        target_velocity=target_velocity,
        guidance_scale=guidance_scale,
        guidance_mode=guidance_mode
    )
    return guidance
