import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.physics_losses import anchor_loss, physics_loss, physics_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Post-optimize generated XYZ motions with lightweight physics losses.")
    parser.add_argument("--results_path", required=True, help="Input MDM results.npy file.")
    parser.add_argument("--output_dir", default="", help="Output directory. Defaults to '<input_dir>_physopt'.")
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--lambda_ground", default=1.0, type=float)
    parser.add_argument("--lambda_foot", default=1.0, type=float)
    parser.add_argument("--lambda_smooth", default=0.05, type=float)
    parser.add_argument("--lambda_anchor", default=1.0, type=float)
    parser.add_argument("--floor_height", default=0.0, type=float)
    parser.add_argument("--contact_height_threshold", default=0.05, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_results(path):
    payload = np.load(path, allow_pickle=True).item()
    motion = torch.from_numpy(payload["motion"]).float()
    lengths = payload.get("lengths")
    if lengths is not None:
        lengths = torch.from_numpy(np.asarray(lengths)).long()
    return payload, motion, lengths


def main():
    args = parse_args()
    payload, motion, lengths = load_results(args.results_path)

    output_dir = args.output_dir
    if output_dir == "":
        output_dir = os.path.dirname(args.results_path).rstrip(os.sep) + "_physopt"
    if os.path.exists(output_dir):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    device = torch.device(args.device)
    original = motion.to(device)
    optimized = original.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([optimized], lr=args.lr)
    lengths_device = lengths.to(device) if lengths is not None else None

    before_metrics = physics_metrics(
        original,
        lengths=lengths_device,
        floor_height=args.floor_height,
        contact_height_threshold=args.contact_height_threshold,
    )

    for _ in trange(args.steps, desc="physics opt"):
        phys, terms = physics_loss(
            optimized,
            lengths=lengths_device,
            lambda_ground=args.lambda_ground,
            lambda_foot=args.lambda_foot,
            lambda_smooth=args.lambda_smooth,
            floor_height=args.floor_height,
            contact_height_threshold=args.contact_height_threshold,
        )
        keep_close = anchor_loss(optimized, original, lengths=lengths_device)
        loss = phys + args.lambda_anchor * keep_close

        opt.zero_grad()
        loss.backward()
        opt.step()

    optimized_np = optimized.detach().cpu().numpy()
    payload["motion"] = optimized_np
    payload["physics_optimization"] = {
        "steps": args.steps,
        "lr": args.lr,
        "lambda_ground": args.lambda_ground,
        "lambda_foot": args.lambda_foot,
        "lambda_smooth": args.lambda_smooth,
        "lambda_anchor": args.lambda_anchor,
        "floor_height": args.floor_height,
        "contact_height_threshold": args.contact_height_threshold,
    }

    out_path = os.path.join(output_dir, "results.npy")
    np.save(out_path, payload)
    for suffix in [".txt", "_len.txt"]:
        src = args.results_path.replace(".npy", suffix)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, os.path.basename(src)))

    after_metrics = physics_metrics(
        torch.from_numpy(optimized_np).float(),
        lengths=lengths,
        floor_height=args.floor_height,
        contact_height_threshold=args.contact_height_threshold,
    )
    metrics = {"before": before_metrics, "after": after_metrics}
    metrics_path = os.path.join(output_dir, "physics_optimization_metrics.json")
    with open(metrics_path, "w") as fw:
        json.dump(metrics, fw, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved optimized results to {out_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
