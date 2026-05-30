# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    physics_neg_dir = getattr(args, 'physics_neg_dir', '')
    physics_pos_dir = getattr(args, 'physics_pos_dir', '')
    if physics_neg_dir and physics_pos_dir:
        from data_loaders.get_data import get_physics_dataset_loader
        print(f'[PhysicsCFG] neg={physics_neg_dir}  pos={physics_pos_dir}')
        data = get_physics_dataset_loader(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            neg_dir=physics_neg_dir,
            pos_dir=physics_pos_dir,
            pos_ratio=getattr(args, 'phys_pos_ratio', 0.5),
            phys_mask_prob=getattr(args, 'phys_mask_prob', 0.1),
            device=dist_util.dev(),
        )
    elif getattr(args, 'vel_cond', False):
        # Velocity CFG training: load GT motions via PhysicsPairedDataset (no GloVe needed).
        # pos_ratio=1.0 → always use GT (new_joint_vecs/) as the training motions.
        # neg_dir is still needed for sample-ID filtering but never loaded in __getitem__.
        from data_loaders.get_data import get_physics_dataset_loader
        neg_dir = getattr(args, 'physics_neg_dir', 'dataset/mdm_negatives')
        print(f'[VelCFG] Loading GT motions via PhysicsPairedDataset (neg_dir={neg_dir})')
        data = get_physics_dataset_loader(
            name=args.dataset,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            neg_dir=neg_dir,
            pos_dir='',          # not used when gt_dir is auto-detected
            pos_ratio=1.0,       # always use GT
            phys_mask_prob=0.0,  # no phys_flag masking (we use vel_cond instead)
            device=dist_util.dev(),
        )
    else:
        data = get_dataset_loader(name=args.dataset,
                                  batch_size=args.batch_size,
                                  num_frames=args.num_frames,
                                  fixed_len=args.pred_len + args.context_len,
                                  pred_len=args.pred_len,
                                  device=dist_util.dev())

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    # Optional: freeze everything except embed_phys_flag for targeted training
    if getattr(args, 'phys_flag_only', False):
        frozen = 0
        for name, p in model.named_parameters():
            if 'embed_phys_flag' not in name and not name.startswith('clip_model.'):
                p.requires_grad_(False)
                frozen += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[phys_flag_only] Frozen {frozen/1e6:.2f}M params. '
              f'Training only {trainable/1e3:.1f}K params (embed_phys_flag).')

    # Optional: freeze everything except embed_vel / vel_null_emb for velocity CFG training
    if getattr(args, 'vel_flag_only', False):
        vel_params = {'embed_vel.weight', 'embed_vel.bias', 'vel_null_emb'}
        frozen = 0
        for name, p in model.named_parameters():
            if name not in vel_params and not name.startswith('clip_model.'):
                p.requires_grad_(False)
                frozen += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[vel_flag_only] Frozen {frozen/1e6:.2f}M params. '
              f'Training only {trainable/1e3:.1f}K params (embed_vel + vel_null_emb).')

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
