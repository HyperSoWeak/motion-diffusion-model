from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate, physics_t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, cache_path=None): 
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader


def get_physics_dataset_loader(name, batch_size, num_frames,
                                neg_dir: str, pos_dir: str,
                                split='train', fixed_len=0,
                                pos_ratio=0.5, phys_mask_prob=0.1, device=None):
    """
    DataLoader for Physics CFG training.
    Uses PhysicsPairedDataset which reads GT + MDM-generated motions directly,
    without requiring GloVe embeddings.
    """
    from data_loaders.humanml.data.physics_paired_dataset import PhysicsPairedDataset
    import numpy as np

    if name not in ['humanml', 'kit']:
        raise ValueError(f'Physics CFG training only supports humanml/kit, got {name}')

    data_root = 'dataset/HumanML3D' if name == 'humanml' else 'dataset/KIT'
    mean = np.load(f'{data_root}/Mean.npy')
    std  = np.load(f'{data_root}/Std.npy')

    dataset = PhysicsPairedDataset(
        data_root=data_root,
        neg_dir=neg_dir,
        pos_dir=pos_dir,
        mean=mean,
        std=std,
        split=split,
        max_frames=num_frames or 196,
        pos_ratio=pos_ratio,
        phys_mask_prob=phys_mask_prob,
    )

    # 設置 mean_gpu / std_gpu 供 training_loop 的 physics loss 使用
    import torch
    dev = device or 'cpu'
    dataset.mean_gpu = torch.tensor(mean, dtype=torch.float32).to(dev)[None, :, None, None]
    dataset.std_gpu  = torch.tensor(std,  dtype=torch.float32).to(dev)[None, :, None, None]

    collate_fn = lambda x: physics_t2m_collate(x, batch_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, collate_fn=collate_fn
    )
    return loader