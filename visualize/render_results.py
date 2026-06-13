"""Render stick-figure videos from results.npy files into one output folder.

Usage:
    python -m visualize.render_results [--out_dir save/all_test_videos] [--num_samples 3]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from moviepy.editor import clips_array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion


def render_one(results_path: str, out_path: str, dataset: str, fps: float, num_vis: int) -> None:
    data = np.load(results_path, allow_pickle=True).item()
    all_motions = data["motion"]   # (n_samples*n_reps, n_joints, 3, n_frames)
    all_text = data["text"]
    all_lengths = data["lengths"]
    n_samples = int(data["num_samples"])
    n_reps = int(data["num_repetitions"])

    skeleton = (
        paramUtil.t2m_kinematic_chain if dataset == "humanml"
        else paramUtil.kit_kinematic_chain
    )
    max_length = int(max(all_lengths))
    num_vis = min(n_samples, num_vis)

    animations = np.empty(shape=(num_vis, n_reps), dtype=object)
    for sample_i in range(num_vis):
        for rep_i in range(n_reps):
            idx = rep_i * n_samples + sample_i
            caption = all_text[idx]
            length = int(all_lengths[idx])
            motion = all_motions[idx].transpose(2, 0, 1)[:max_length]
            if motion.shape[0] > length:
                motion[length:] = motion[length - 1]

            # save_path is unused internally; clip is returned in memory
            animations[sample_i, rep_i] = plot_3d_motion(
                out_path, skeleton, motion, dataset=dataset, title=caption, fps=fps
            )

    clips = clips_array(animations)
    clips.duration = max_length / fps
    clips.write_videofile(out_path, fps=fps, threads=4, logger=None)
    for clip in clips.clips:
        clip.close()
    clips.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="save/all_test_videos")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples per test to include in the grid")
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--dataset", default="humanml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results_dirs = sorted(
        str(p.parent) for p in Path("save").rglob("results.npy")
    )
    print(f"Found {len(results_dirs)} directories with results.npy")

    for results_dir in results_dirs:
        rel = os.path.relpath(results_dir, "save")
        name = rel.replace("/", "__").replace("\\", "__")
        out_file = os.path.join(args.out_dir, f"{name}.mp4")

        if os.path.exists(out_file):
            print(f"[skip] {name}")
            continue

        print(f"\n=== {name} ===")
        try:
            render_one(
                os.path.join(results_dir, "results.npy"),
                out_file,
                dataset=args.dataset,
                fps=args.fps,
                num_vis=args.num_samples,
            )
        except Exception as exc:
            print(f"[error] {name}: {exc}")

    print(f"\nDone. Videos saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
