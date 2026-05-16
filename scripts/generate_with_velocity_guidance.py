#!/usr/bin/env python
"""
Example script demonstrating velocity-guided motion generation.

This script shows how to use the velocity guidance feature to control
motion trajectories by specifying target root velocities.
"""

import os
import sys
import torch
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from utils.velocity_guidance_sampler import VelocityGuidedSampleModel
from model.velocity_classifier import VelocityRegressor
from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import argparse
from utils.fixseed import fixseed


def generate_with_velocity_guidance(
    model_path,
    text_prompt,
    target_velocity=(1.0, 0.0),
    guidance_scale=0.3,
    num_samples=1,
    num_repetitions=1,
    seed=42,
    output_dir=None
):
    """
    Generate motions with velocity guidance.
    
    Args:
        model_path: Path to the trained MDM model
        text_prompt: Text description of the motion
        target_velocity: Tuple (vx, vz) specifying target root velocity
        guidance_scale: Strength of velocity guidance (0-1)
        num_samples: Number of samples to generate
        num_repetitions: Number of repetitions per sample
        seed: Random seed
        output_dir: Output directory for generated samples
    
    Returns:
        motions: Generated motion samples (B, C, T)
        texts: Corresponding text prompts
    """
    
    fixseed(seed)
    dist_util.setup_dist('cuda' if torch.cuda.is_available() else 'cpu')
    
    if output_dir is None:
        output_dir = './velocity_guided_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup dataset and dataloader
    max_frames = 196
    n_frames = int(6.0 * 20)  # 6 seconds at 20 fps
    
    print('Loading dataset...')
    data = get_dataset_loader(
        name='humanml',
        batch_size=num_samples,
        num_frames=n_frames,
        split='test',
        hml_mode='text_only'
    )
    
    print('Creating model and diffusion...')
    args = argparse.Namespace(
        dataset='humanml',
        model_path=model_path,
        diffusion_steps=1000,
        guidance_param=1.0,  # Will be set later if using text CFG
        cond_mask_prob=0.1,
        use_ema=True
    )
    
    model, diffusion = create_model_and_diffusion(args, data)
    
    print(f'Loading model from {model_path}...')
    load_saved_model(model, model_path, use_avg=True)
    
    # Create velocity regressor
    print('Initializing velocity regressor...')
    input_feats = model.njoints * model.nfeats
    velocity_regressor = VelocityRegressor(
        input_feats=input_feats,
        latent_dim=256,
        num_layers=3,
        dropout=0.1
    )
    velocity_regressor.to(dist_util.dev())
    velocity_regressor.eval()
    
    # Wrap model with velocity guidance
    print('Wrapping model with velocity guidance...')
    model = VelocityGuidedSampleModel(
        model=model,
        velocity_regressor=velocity_regressor,
        use_classifier_free=False
    )
    
    # Set velocity guidance parameters
    model.velocity_guidance_scale = guidance_scale
    model.velocity_guidance_mode = 'direction'
    if target_velocity:
        model.target_velocity = torch.tensor(
            [list(target_velocity)] * num_samples,
            device=dist_util.dev(),
            dtype=torch.float32
        )
    
    model.to(dist_util.dev())
    model.eval()
    
    # Prepare text conditioning
    collate_args = []
    for i in range(num_samples):
        collate_args.append({
            'inp': torch.zeros(n_frames),
            'tokens': None,
            'lengths': n_frames,
            'text': text_prompt
        })
    
    _, model_kwargs = collate(collate_args)
    model_kwargs['y'] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs['y'].items()
    }
    
    # Encode text
    if hasattr(model, 'encode_text'):
        model_kwargs['y']['text_embed'] = model.encode_text(
            model_kwargs['y']['text']
        )
    
    # Generate motions
    print(f'Generating {num_samples} motions with velocity guidance...')
    print(f'  Target velocity: {target_velocity}')
    print(f'  Guidance scale: {guidance_scale}')
    print(f'  Text prompt: "{text_prompt}"')
    
    motion_shape = (num_samples, model.njoints, model.nfeats, n_frames)
    
    sample_fn = diffusion.p_sample_loop
    
    sample = sample_fn(
        model,
        motion_shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    
    # Recover XYZ positions
    print('Recovering motion representations...')
    n_joints = 22 if sample.shape[1] == 263 else 21
    sample = data.dataset.t2m_dataset.inv_transform(
        sample.cpu().permute(0, 2, 3, 1)
    ).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    
    print(f'Generated motion shape: {sample.shape}')
    print(f'Motion saved to {output_dir}')
    
    return sample, [text_prompt] * num_samples


def main():
    """Run example velocity-guided generation."""
    
    print("="*60)
    print("Velocity-Guided Motion Generation Example")
    print("="*60)
    
    # Example 1: Forward walking
    print("\n[Example 1] Generating forward-walking motion...")
    motions_1, texts_1 = generate_with_velocity_guidance(
        model_path='path/to/humanml_final.pt',
        text_prompt='a person walks forward',
        target_velocity=(1.0, 0.0),  # Forward
        guidance_scale=0.3,
        num_samples=2,
        output_dir='./samples/forward_walk'
    )
    
    # Example 2: Diagonal motion
    print("\n[Example 2] Generating diagonal-motion...")
    motions_2, texts_2 = generate_with_velocity_guidance(
        model_path='path/to/humanml_final.pt',
        text_prompt='a person moves diagonally',
        target_velocity=(0.7, 0.7),  # Forward-right
        guidance_scale=0.25,
        num_samples=2,
        output_dir='./samples/diagonal_move'
    )
    
    # Example 3: Fast motion
    print("\n[Example 3] Generating fast motion...")
    motions_3, texts_3 = generate_with_velocity_guidance(
        model_path='path/to/humanml_final.pt',
        text_prompt='a person runs',
        target_velocity=(1.2, 0.0),  # Fast forward
        guidance_scale=0.4,
        num_samples=2,
        output_dir='./samples/fast_run'
    )
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
