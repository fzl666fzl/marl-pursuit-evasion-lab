from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pursuit_lab.metrics import write_csv


def find_run_dirs(runs_dir: str | Path) -> list[Path]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def plot_training_curve(run_dirs: list[Path], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists() or metrics_path.stat().st_size == 0:
            continue
        metrics = pd.read_csv(metrics_path)
        if {"episode", "episode_reward"}.issubset(metrics.columns):
            rewards = metrics["episode_reward"].rolling(window=25, min_periods=1).mean()
            ax.plot(metrics["episode"], rewards, label=run_dir.name)
            plotted = True

    ax.set_title("Training Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean pursuer reward")
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No training metrics found", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_eval_comparison(run_dirs: list[Path], output_path: Path) -> None:
    rows = comparison_rows(run_dirs)
    names = [row["experiment"] for row in rows]
    capture_rates = [float(row["capture_rate"]) for row in rows]
    steps = [float(row["mean_steps_to_capture"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    if names:
        positions = range(len(names))
        ax1.bar(positions, capture_rates, color="#4c78a8", label="capture_rate")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Capture rate")
        ax1.set_xticks(list(positions), names, rotation=20, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(list(positions), steps, color="#f58518", marker="o", label="mean_steps_to_capture")
        ax2.set_ylabel("Steps to capture")
    else:
        ax1.text(0.5, 0.5, "No eval summaries found", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xticks([])
    ax1.set_title("Evaluation Comparison")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def comparison_rows(run_dirs: list[Path]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "eval_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": run_dir.name,
                "episodes": int(summary.get("episodes", 0)),
                "capture_rate": float(summary.get("capture_rate", 0.0)),
                "mean_episode_reward": float(summary.get("mean_episode_reward", 0.0)),
                "mean_steps_to_capture": float(summary.get("mean_steps_to_capture", 0.0)),
                "success_episode_count": int(summary.get("success_episode_count", 0)),
            }
        )
    return rows


def create_plots(runs_dir: str | Path, figures_dir: str | Path) -> tuple[Path, Path, Path]:
    run_dirs = find_run_dirs(runs_dir)
    figures = Path(figures_dir)
    training_path = figures / "training_curve.png"
    comparison_path = figures / "eval_comparison.png"
    comparison_csv_path = figures / "eval_comparison.csv"
    plot_training_curve(run_dirs, training_path)
    plot_eval_comparison(run_dirs, comparison_path)
    write_csv(comparison_rows(run_dirs), comparison_csv_path)
    return training_path, comparison_path, comparison_csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot pursuit_lab training and evaluation outputs.")
    parser.add_argument("--runs", default="runs", help="Runs directory.")
    parser.add_argument("--figures", default="figures", help="Output figures directory.")
    args = parser.parse_args(argv)

    paths = create_plots(args.runs, args.figures)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
