from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pursuit_lab.constants import PURSUER_AGENTS
from pursuit_lab.dqn import DQN, select_epsilon_greedy_action
from pursuit_lab.envs import make_env
from pursuit_lab.metrics import summarize_episodes, write_csv, write_json


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_policy_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    if checkpoint.get("algorithm") == "mappo":
        from pursuit_lab.mappo import MAPPOActor

        model = MAPPOActor(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_sizes=tuple(checkpoint["config"]["mappo"]["actor_hidden_sizes"]),
        )
        model.load_state_dict(checkpoint["actor_state_dict"])
        model.eval()
        return model

    model = DQN(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_sizes=tuple(checkpoint["config"]["dqn"]["hidden_sizes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def episode_seed(seeds: list[int], episode_index: int) -> int:
    if not seeds:
        return episode_index
    cycle = episode_index // len(seeds)
    return int(seeds[episode_index % len(seeds)]) + cycle * 1000


def run_eval_episodes(
    *,
    policy_net: nn.Module | None,
    env_config: dict[str, Any],
    episodes: int,
    seeds: list[int],
    render_mode: str | None = None,
    collect_frames: bool = False,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    max_cycles = int(env_config.get("max_cycles", 50))

    for episode in range(episodes):
        seed = episode_seed(seeds, episode)
        rng = np.random.default_rng(seed)
        env = make_env(seed=seed, render_mode=render_mode, env_config=env_config)
        captured = False
        episode_reward = 0.0
        steps = 0
        try:
            observations, _ = env.reset(seed=seed)
            if collect_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            for step in range(max_cycles):
                actions: dict[str, int] = {}
                for agent in env.agents:
                    action_space = env.action_space(agent)
                    if policy_net is not None and agent in PURSUER_AGENTS:
                        actions[agent] = select_epsilon_greedy_action(
                            network=policy_net,
                            observation=observations[agent],
                            epsilon=0.0,
                            action_dim=int(action_space.n),
                            rng=rng,
                        )
                    else:
                        actions[agent] = int(action_space.sample())

                observations, rewards, terminations, truncations, _ = env.step(actions)
                episode_reward += sum(float(rewards.get(agent, 0.0)) for agent in PURSUER_AGENTS)
                steps = step + 1
                captured = captured or any(bool(value) for value in terminations.values())

                if collect_frames:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                if all(terminations.values()) or all(truncations.values()) or not env.agents:
                    break
        finally:
            env.close()

        rows.append(
            {
                "episode": episode + 1,
                "seed": seed,
                "captured": captured,
                "steps": steps,
                "episode_reward": episode_reward,
            }
        )

    return rows, frames


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int,
    output_dir: str | Path | None = None,
    seeds: list[int] | None = None,
) -> dict[str, float | int]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    policy = build_policy_from_checkpoint(checkpoint)
    eval_seeds = list(seeds if seeds is not None else config["evaluation"]["seeds"])
    rows, _ = run_eval_episodes(
        policy_net=policy,
        env_config=config["env"],
        episodes=episodes,
        seeds=eval_seeds,
    )
    summary = summarize_episodes(rows, max_cycles=int(config["env"].get("max_cycles", 50)))

    if output_dir is not None:
        target = Path(output_dir)
        write_csv(rows, target / "eval_episodes.csv")
        write_json(summary, target / "eval_summary.json")
    return summary
