from __future__ import annotations

import argparse
from pathlib import Path

from pursuit_lab.evaluation import evaluate_checkpoint
from pursuit_lab.metrics import write_csv


def find_checkpoint_runs(runs_dir: str | Path) -> list[Path]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "best.pt").is_file())


def evaluate_run_checkpoints(
    *,
    runs_dir: str | Path,
    output_dir: str | Path,
    episodes: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    output_root = Path(output_dir)
    for run_dir in find_checkpoint_runs(runs_dir):
        experiment = run_dir.name
        summary = evaluate_checkpoint(
            checkpoint_path=run_dir / "best.pt",
            episodes=episodes,
            output_dir=output_root / experiment,
        )
        rows.append(
            {
                "experiment": experiment,
                "episodes": int(summary["episodes"]),
                "capture_rate": float(summary["capture_rate"]),
                "mean_episode_reward": float(summary["mean_episode_reward"]),
                "mean_steps_to_capture": float(summary["mean_steps_to_capture"]),
                "success_episode_count": int(summary["success_episode_count"]),
            }
        )
    write_csv(rows, output_root / "eval_comparison.csv")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate best.pt checkpoints under a runs directory.")
    parser.add_argument("--runs", required=True, help="Runs directory containing experiment subdirectories.")
    parser.add_argument("--output", required=True, help="Directory for per-run eval outputs and comparison CSV.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes per checkpoint.")
    args = parser.parse_args(argv)

    rows = evaluate_run_checkpoints(
        runs_dir=args.runs,
        output_dir=args.output,
        episodes=args.episodes,
    )
    for row in rows:
        print(
            f"{row['experiment']}: capture_rate={row['capture_rate']}, "
            f"episodes={row['episodes']}, successes={row['success_episode_count']}"
        )


if __name__ == "__main__":
    main()
