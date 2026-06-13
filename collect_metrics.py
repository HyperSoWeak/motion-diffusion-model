#!/usr/bin/env python3
"""Collect ablation physics_metrics.json -> JSON + Markdown report."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

# 位元順序 [g][f][s][r]
RUNS = [
    ("ablate_0000", "baseline (g,f,s,r off)"),
    ("ablate_1000", "ground only"),
    ("ablate_0100", "foot only"),
    ("ablate_0010", "smooth only"),
    ("ablate_1100", "ground + foot"),
    ("ablate_1010", "ground + smooth"),
    ("ablate_0110", "foot + smooth"),
    ("ablate_1110", "ground + foot + smooth (r=0)"),
]

# 報告主表（與 physics_residual_full_experiment_report 32-prompt 表一致）
TABLE_KEYS = [
    ("foot_skating_score", "Foot skating"),
    ("ground_penetration_rate", "Ground penetration rate"),
    ("mean_penetration_depth", "Mean penetration"),
    ("max_penetration_depth", "Max penetration"),
    ("mean_acceleration", "Mean acceleration"),
]

# 可選：含 foot_contact_rate 的擴充表
EXTRA_KEYS = [
    ("foot_contact_rate", "Foot contact rate"),
]

LOWER_BETTER = {k for k, _ in TABLE_KEYS}


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_val(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 0.01:
        return f"{v:.5f}"
    if av >= 1e-4:
        return f"{v:.8f}"
    return f"{v:.3e}"


def pct_change(baseline: float, value: float) -> str:
    if baseline == 0:
        return "n/a"
  # 負數 = 改善（lower is better）
    return f"{(value - baseline) / baseline * 100:+.1f}%"


def best_run_id(rows: list[dict], key: str) -> str:
    eligible = [r for r in rows if r.get(key) is not None]
    if not eligible:
        return "n/a"
    if key in LOWER_BETTER:
        return min(eligible, key=lambda r: r[key])["run_id"]
    return max(eligible, key=lambda r: r[key])["run_id"]


def collect(eval_root: Path, suffix: str) -> list[dict]:
    rows = []
    for run_id, label in RUNS:
        metrics_path = eval_root / f"{run_id}{suffix}" / "physics_metrics.json"
        m = load_metrics(metrics_path)
        rows.append({
            "run_id": run_id,
            "label": label,
            "metrics_path": str(metrics_path),
            **m,
        })
    return rows


def write_markdown(
    out_md: Path,
    rows: list[dict],
    baseline_id: str,
    eval_root: Path,
    suffix: str,
    include_contact: bool,
) -> None:
    baseline = next(r for r in rows if r["run_id"] == baseline_id)
    keys = TABLE_KEYS + (EXTRA_KEYS if include_contact else [])

    lines = [
        "# Physics Ablation Report (g,f,s,r — root=0)",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Summary",
        "",
        "Collected from `eval/eval_physics.py` on generated `results.npy`.",
        "",
        "## Evaluation Setup",
        "",
        "```text",
        f"eval_root:     {eval_root}",
        f"sample_dirs:  {{run_id}}{suffix}/",
        "prompts:       reports/physics_eval_prompts_32.txt",
        "sampling:      32 prompts x 2 repetitions = 64 motions (if generate used defaults)",
        "guidance_param: 2.5 (if generate used default)",
        "```",
        "",
        "## Physics Metrics",
        "",
        "Lower is better for all metrics below except `foot_contact_rate`, which is diagnostic.",
        "",
    ]

    header = "| Run | Label | " + " | ".join(h for _, h in keys) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(keys)) + "|"
    lines.extend([header, sep])
    for r in rows:
        vals = [fmt_val(r[k]) for k, _ in keys]
        lines.append(f"| {r['run_id']} | {r['label']} | " + " | ".join(vals) + " |")

    lines.extend([
        "",
        "## Relative to baseline (`" + baseline_id + "`)",
        "",
        "Negative % means improvement (metric decreased).",
        "",
        "| Run | " + " | ".join(h for _, h in TABLE_KEYS) + " |",
        "|---|" + "|".join(["---:"] * len(TABLE_KEYS)) + "|",
    ])
    for r in rows:
        if r["run_id"] == baseline_id:
            continue
        rel = [pct_change(baseline[k], r[k]) for k, _ in TABLE_KEYS]
        lines.append(f"| {r['run_id']} | " + " | ".join(rel) + " |")

    lines.extend(["", "## Best per metric", "", "```text"])
    for k, label in TABLE_KEYS:
        lines.append(f"{label}: {best_run_id(rows, k)}")
    lines.append("```")
    lines.append("")

    missing = [r["run_id"] for r in rows if not Path(r["metrics_path"]).exists()]
    if missing:
        lines.extend([
            "## Missing",
            "",
            "The following runs have no `physics_metrics.json`:",
            "",
            "```text",
            *missing,
            "```",
            "",
        ])

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path("save/physics_ablate32"),
        help="Root containing ablate_XXXX_samples/",
    )
    p.add_argument(
        "--suffix",
        default="_samples",
        help='Subdir suffix, e.g. "_samples" or "_samples_physopt"',
    )
    p.add_argument("--baseline", default="ablate_0000")
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/physics_ablate_gfsr0_metrics.json"),
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/physics_ablate_gfsr0_report.md"),
    )
    p.add_argument("--include-contact", action="store_true")
    args = p.parse_args()

    rows = collect(args.eval_root, args.suffix)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(
        args.out_md, rows, args.baseline, args.eval_root, args.suffix, args.include_contact
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
