"""
generate_physics_negatives.py
用原始 MDM（無任何物理 guidance）對 HumanML3D training set 的每個
text prompt 生成一筆動作，存成 [T, 263] npy，作為 Physics CFG training
的 physics_flag=0 負樣本。

輸出目錄結構：
  {output_dir}/
    {sample_id}.npy      # [T, 263] 正規化 hml_vec，跟 GT 格式相同
    meta.json            # {sample_id: {"text": ..., "length": T}}
    done.txt             # 完成後建立，可用來 resume

用法：
  python -m scripts.generate_physics_negatives \
      --model_path save/humanml_enc_512_50steps/model000750000.pt \
      --output_dir dataset/mdm_negatives \
      --batch_size 32 \
      --device 0

Resume：如果中途中斷，重新執行同樣指令，已存在的 sample 會跳過。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ── 讓 repo root 可以 import ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.tensors import collate


# ── 讀 HumanML3D train split ──────────────────────────────────────────────────

def load_train_ids(split_file: str) -> list[str]:
    """讀 train.txt，去掉 M 前綴，回傳去重後的 ID 列表（保持順序）。"""
    ids = []
    seen = set()
    with open(split_file) as f:
        for line in f:
            sid = line.strip().lstrip('M')
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def read_first_text(text_dir: str, sample_id: str) -> str | None:
    """讀 texts/{sample_id}.txt 的第一行，取 # 分隔的第一個欄位（純文字）。"""
    path = os.path.join(text_dir, f'{sample_id}.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    return line.split('#')[0].strip()


# ── 從模型輸出取 hml_vec ──────────────────────────────────────────────────────

def output_to_hmlvec(sample: torch.Tensor) -> np.ndarray:
    """
    sample : [bs, 263, 1, T]  正規化 hml_vec（模型原始輸出）
    returns: [T, 263] float32 numpy array（跟 GT new_joint_vecs 格式相同）
    """
    # [bs, 263, 1, T] → [263, T] → [T, 263]
    return sample[0, :, 0, :].cpu().float().numpy().T


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', default='dataset/mdm_negatives')
    parser.add_argument('--split_file',
                        default='dataset/HumanML3D/train.txt')
    parser.add_argument('--text_dir',
                        default='dataset/HumanML3D/texts')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--guidance_param', type=float, default=2.5)
    parser.add_argument('--motion_length', type=float, default=6.0,
                        help='每筆生成的秒數（決定 nframes = length × 20）')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, 'meta.json')
    done_path = os.path.join(args.output_dir, 'done.txt')

    # 已完成的 meta（resume 用）
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    dist_util.setup_dist(args.device)

    # ── 建立 fake args 給 create_model_and_diffusion ────────────────────────
    # 直接借用 generate_args() 的邏輯，但只需要最小參數
    from utils.parser_util import generate_args
    import sys as _sys
    _sys.argv = [
        'generate',
        '--model_path', args.model_path,
        '--num_samples', '1',
        '--device', str(args.device),
    ]
    gen_args = generate_args()
    gen_args.batch_size = args.batch_size

    from data_loaders.get_data import get_dataset_loader
    max_frames = 196
    fps = 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    print('Loading dataset (for mean/std and dataloader)...')
    data = get_dataset_loader(
        name='humanml',
        batch_size=args.batch_size,
        num_frames=max_frames,
        split='train',
        hml_mode='text_only',
    )
    data.fixed_length = n_frames

    print('Creating model and diffusion...')
    model, diffusion = create_model_and_diffusion(gen_args, data)
    load_saved_model(model, args.model_path, use_avg=gen_args.use_ema)

    if gen_args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    sample_fn = diffusion.p_sample_loop

    # ── 讀所有 training IDs ─────────────────────────────────────────────────
    all_ids = load_train_ids(args.split_file)
    print(f'Total unique IDs: {len(all_ids)}')

    # 過濾掉已完成的
    todo_ids = [sid for sid in all_ids
                if sid not in meta
                and not os.path.exists(os.path.join(args.output_dir, f'{sid}.npy'))]
    print(f'Remaining: {len(todo_ids)}  (skipping {len(all_ids) - len(todo_ids)} done)')

    # ── 批次生成 ────────────────────────────────────────────────────────────
    skipped = 0
    for batch_start in tqdm(range(0, len(todo_ids), args.batch_size),
                            desc='Generating'):
        batch_ids = todo_ids[batch_start: batch_start + args.batch_size]

        # 讀文字；跳過找不到 text 的 ID
        texts, valid_ids = [], []
        for sid in batch_ids:
            txt = read_first_text(args.text_dir, sid)
            if txt is None:
                skipped += 1
                continue
            texts.append(txt)
            valid_ids.append(sid)

        if not texts:
            continue

        bs = len(texts)
        collate_args = [
            {'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames,
             'text': txt}
            for txt in texts
        ]
        _, model_kwargs = collate(collate_args)
        model_kwargs['y'] = {
            k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
            for k, v in model_kwargs['y'].items()
        }
        model_kwargs['y']['scale'] = (
            torch.ones(bs, device=dist_util.dev()) * args.guidance_param
        )
        model_kwargs['y']['text_embed'] = model.encode_text(
            model_kwargs['y']['text']
        )

        motion_shape = (bs, model.njoints, model.nfeats, n_frames)

        with torch.no_grad():
            sample = sample_fn(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )   # [bs, 263, 1, n_frames]

        # ── 存檔 ────────────────────────────────────────────────────────────
        for i, sid in enumerate(valid_ids):
            hml = sample[i:i+1]          # [1, 263, 1, n_frames]
            arr = output_to_hmlvec(hml)  # [n_frames, 263]
            npy_path = os.path.join(args.output_dir, f'{sid}.npy')
            np.save(npy_path, arr)
            meta[sid] = {'text': texts[i], 'length': int(arr.shape[0])}

        # 每批次更新 meta（方便 resume）
        with open(meta_path, 'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f'\nDone. Generated {len(meta)} samples. Skipped (no text): {skipped}')
    with open(done_path, 'w') as f:
        f.write(f'total={len(meta)}\n')


if __name__ == '__main__':
    main()
