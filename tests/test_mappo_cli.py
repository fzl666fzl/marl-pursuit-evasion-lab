from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def write_mappo_smoke_config(path: Path, run_dir: Path) -> None:
    config = {
        "experiment_name": "mappo_smoke",
        "algorithm": "mappo",
        "seed": 0,
        "output_dir": str(run_dir),
        "total_episodes": 4,
        "env": {
            "num_good": 1,
            "num_adversaries": 3,
            "num_obstacles": 2,
            "max_cycles": 20,
            "continuous_actions": False,
            "terminate_on_success": True,
        },
        "mappo": {
            "actor_hidden_sizes": [32],
            "critic_hidden_sizes": [32],
            "learning_rate": 0.001,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "update_epochs": 2,
            "rollout_episodes": 2,
            "max_grad_norm": 0.5,
        },
        "evaluation": {"eval_interval": 2, "eval_episodes": 2, "seeds": [0]},
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


def test_mappo_train_evaluate_and_render_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "mappo.yaml"
    write_mappo_smoke_config(config_path, runs_dir)

    run_module(
        [
            "pursuit_lab.train",
            "--config",
            str(config_path),
        ],
        cwd=repo,
    )

    run_dir = runs_dir / "mappo_smoke"
    checkpoint = run_dir / "best.pt"

    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["algorithm"] == "mappo"
    assert "actor_state_dict" in saved
    assert "critic_state_dict" in saved
    assert (run_dir / "metrics.csv").is_file()
    with (run_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4

    run_module(
        [
            "pursuit_lab.evaluate",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "2",
            "--output",
            str(run_dir),
        ],
        cwd=repo,
    )
    assert (run_dir / "eval_summary.json").is_file()

    gif_path = tmp_path / "videos" / "mappo.gif"
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
