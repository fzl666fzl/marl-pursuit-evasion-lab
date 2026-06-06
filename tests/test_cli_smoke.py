from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


def write_smoke_config(path: Path, run_dir: Path) -> None:
    config = {
        "experiment_name": "smoke_dqn",
        "seed": 0,
        "output_dir": str(run_dir),
        "total_episodes": 3,
        "env": {
            "num_good": 1,
            "num_adversaries": 3,
            "num_obstacles": 2,
            "max_cycles": 10,
            "continuous_actions": False,
            "terminate_on_success": True,
        },
        "dqn": {"hidden_sizes": [32], "learning_rate": 0.001, "gamma": 0.95},
        "training": {
            "batch_size": 8,
            "buffer_capacity": 200,
            "min_buffer_size": 8,
            "target_update_interval": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_episodes": 3,
            "team_reward_weight": 0.0,
            "curriculum": False,
        },
        "evaluation": {"eval_interval": 3, "eval_episodes": 1, "seeds": [0]},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def run_module(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def test_train_evaluate_render_and_plot_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "smoke.yaml"
    write_smoke_config(config_path, runs_dir)

    run_module(
        [
            "pursuit_lab.train",
            "--config",
            str(config_path),
            "--episodes",
            "3",
            "--experiment-name",
            "smoke_override",
            "--output-dir",
            str(runs_dir),
        ],
        cwd=repo,
    )
    run_dir = runs_dir / "smoke_override"
    checkpoint = run_dir / "best.pt"

    assert (run_dir / "metrics.csv").is_file()
    assert checkpoint.is_file()
    assert (run_dir / "config.yaml").is_file()
    with (run_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3

    run_module(
        [
            "pursuit_lab.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "1",
            "--output",
            str(run_dir),
        ],
        cwd=repo,
    )
    assert (run_dir / "eval_summary.json").is_file()
    assert (run_dir / "eval_episodes.csv").is_file()

    gif_path = tmp_path / "videos" / "pursuit_demo.gif"
    run_module(
        [
            "pursuit_lab.render",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "1",
            "--output",
            str(gif_path),
        ],
        cwd=repo,
    )
    assert gif_path.is_file()
    assert gif_path.stat().st_size > 0

    figures_dir = tmp_path / "figures"
    run_module(
        [
            "pursuit_lab.plot",
            "--runs",
            str(runs_dir),
            "--figures",
            str(figures_dir),
        ],
        cwd=repo,
    )
    assert (figures_dir / "training_curve.png").is_file()
    assert (figures_dir / "eval_comparison.png").is_file()
    comparison_csv = figures_dir / "eval_comparison.csv"
    assert comparison_csv.is_file()
    with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experiment"] == "smoke_override"
    assert "capture_rate" in rows[0]
