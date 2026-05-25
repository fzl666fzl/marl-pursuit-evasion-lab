from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pursuit_lab.config import load_config
from pursuit_lab.mappo import train_mappo
from pursuit_lab.plot import create_plots
from pursuit_lab.train import run_random_baseline, train_dqn

DEFAULT_CONFIGS = (
    "configs/random_baseline.yaml",
    "configs/dqn_baseline.yaml",
    "configs/dqn_curriculum_team_reward.yaml",
    "configs/mappo_baseline.yaml",
)


def apply_overrides(
    config: dict[str, Any],
    *,
    output_dir: str | None,
    episodes: int | None,
    seed: int | None,
) -> dict[str, Any]:
    if output_dir is not None:
        config["output_dir"] = output_dir
    if episodes is not None and config.get("algorithm") != "random":
        config["total_episodes"] = episodes
    if seed is not None:
        config["seed"] = seed
    return config


def run_configs(
    *,
    config_paths: list[str],
    output_dir: str | None,
    figures_dir: str,
    episodes: int | None,
    seed: int | None,
) -> list[Path]:
    run_dirs: list[Path] = []
    for config_path in config_paths:
        config = apply_overrides(
            load_config(config_path),
            output_dir=output_dir,
            episodes=episodes,
            seed=seed,
        )
        if config.get("algorithm") == "random" or config["experiment_name"] == "random_baseline":
            run_dirs.append(run_random_baseline(config))
        elif config.get("algorithm") == "mappo":
            run_dirs.append(train_mappo(config))
        else:
            run_dirs.append(train_dqn(config))

    runs_root = output_dir or "runs"
    return run_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run multiple pursuit_lab experiments and plot results.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Config files to run in order.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output_dir for all configs.")
    parser.add_argument("--figures", default="figures", help="Output figures directory.")
    parser.add_argument("--episodes", type=int, default=None, help="Override total_episodes for trainable configs.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for all configs.")
    args = parser.parse_args(argv)

    run_dirs = run_configs(
        config_paths=list(args.configs),
        output_dir=args.output_dir,
        figures_dir=args.figures,
        episodes=args.episodes,
        seed=args.seed,
    )
    plot_paths = create_plots(args.output_dir or "runs", args.figures)
    for run_dir in run_dirs:
        print(run_dir)
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
