import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.physics_losses import HML_FOOT_JOINTS, physics_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lightweight physics metrics for MDM results.npy files.")
    parser.add_argument("--results_path", required=True, help="Path to a generated results.npy file.")
    parser.add_argument("--output_json", default="", help="Where to write metrics. Defaults next to results.npy.")
    parser.add_argument("--floor_height", default=0.0, type=float)
    parser.add_argument("--contact_height_threshold", default=0.05, type=float)
    parser.add_argument(
        "--foot_indices",
        default=",".join(str(i) for i in HML_FOOT_JOINTS),
        help="Comma-separated foot joint indices. HumanML3D default: 7,10,8,11.",
    )
    return parser.parse_args()


def load_results(path):
    payload = np.load(path, allow_pickle=True).item()
    if "motion" not in payload:
        raise KeyError(f"{path} does not contain a 'motion' entry")
    motion = torch.from_numpy(payload["motion"]).float()
    lengths = payload.get("lengths")
    if lengths is not None:
        lengths = torch.from_numpy(np.asarray(lengths)).long()
    return payload, motion, lengths


def main():
    args = parse_args()
    _, motion, lengths = load_results(args.results_path)
    foot_indices = tuple(int(idx) for idx in args.foot_indices.split(",") if idx != "")

    metrics = physics_metrics(
        motion,
        lengths=lengths,
        foot_indices=foot_indices,
        floor_height=args.floor_height,
        contact_height_threshold=args.contact_height_threshold,
    )

    output_json = args.output_json
    if output_json == "":
        output_json = os.path.join(os.path.dirname(args.results_path), "physics_metrics.json")
    with open(output_json, "w") as fw:
        json.dump(metrics, fw, indent=2, sort_keys=True)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
