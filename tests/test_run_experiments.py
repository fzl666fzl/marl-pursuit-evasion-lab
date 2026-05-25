from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml


def write_config(path: Path, *, experiment_name: str, algorithm: str | None = None) -> None:
    config = {
        "experiment_name": experiment_name,
        "seed": 0,
        "output_dir": "runs",
        "total_episodes": 3,
        "env": {
            "num_good": 1,
            "num_adversaries": 3,
            "num_obstacles": 2,
            "max_cycles": 50,
            "continuous_actions": False,
            "terminate_on_success": True,
        },
        "dqn": {"hidden_sizes": [32], "learning_rate": 0.001, "gamma": 0.95},
        "training": {
            "batch_size": 4,
            "buffer_capacity": 100,
            "min_buffer_size": 4,
            "target_update_interval": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_episodes": 3,
            "team_reward_weight": 0.0,
            "curriculum": False,
        },
        "evaluation": {"eval_interval": 3, "eval_episodes": 2, "seeds": [0]},
    }
    if algorithm is not None:
        config["algorithm"] = algorithm
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_run_experiments_runs_configs_and_writes_comparison(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    figures_dir = tmp_path / "figures"
    random_config = tmp_path / "random.yaml"
    dqn_config = tmp_path / "dqn.yaml"
    write_config(random_config, experiment_name="random_smoke", algorithm="random")
    write_config(dqn_config, experiment_name="dqn_smoke")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pursuit_lab.run_experiments",
            "--configs",
            str(random_config),
            str(dqn_config),
            "--output-dir",
            str(runs_dir),
            "--figures",
            str(figures_dir),
            "--episodes",
            "3",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (runs_dir / "random_smoke" / "eval_summary.json").is_file()
    assert (runs_dir / "dqn_smoke" / "best.pt").is_file()
    comparison_csv = figures_dir / "eval_comparison.csv"
    assert comparison_csv.is_file()
    with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == ["dqn_smoke", "random_smoke"]


def test_run_experiments_help_describes_episode_override_scope() -> None:
    repo = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "-m", "pursuit_lab.run_experiments", "--help"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Override total_episodes for trainable configs" in completed.stdout
