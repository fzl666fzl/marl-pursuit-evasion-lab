from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from pursuit_lab.constants import DEFAULT_ENV_KWARGS


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "dqn_baseline",
    "seed": 0,
    "output_dir": "runs",
    "total_episodes": 3000,
    "env": DEFAULT_ENV_KWARGS,
    "dqn": {
        "hidden_sizes": [128, 128],
        "learning_rate": 0.001,
        "gamma": 0.95,
    },
    "mappo": {
        "actor_hidden_sizes": [128, 128],
        "critic_hidden_sizes": [128, 128],
        "learning_rate": 0.001,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "update_epochs": 4,
        "rollout_episodes": 10,
        "max_grad_norm": 0.5,
    },
    "training": {
        "batch_size": 64,
        "buffer_capacity": 50000,
        "min_buffer_size": 1000,
        "target_update_interval": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_episodes": 2500,
        "team_reward_weight": 0.0,
        "curriculum": False,
    },
    "evaluation": {
        "eval_interval": 250,
        "eval_episodes": 10,
        "seeds": [0, 1, 2, 3, 4],
    },
}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return deep_merge(DEFAULT_CONFIG, data)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
