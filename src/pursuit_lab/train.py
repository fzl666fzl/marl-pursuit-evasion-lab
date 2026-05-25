from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from pursuit_lab.config import load_config, save_config
from pursuit_lab.constants import PURSUER_AGENTS
from pursuit_lab.dqn import DQN, linear_epsilon, optimize_dqn, select_epsilon_greedy_action
from pursuit_lab.envs import apply_curriculum_stage, curriculum_stage_for_episode, make_env
from pursuit_lab.evaluation import run_eval_episodes
from pursuit_lab.metrics import summarize_episodes, write_csv, write_json
from pursuit_lab.mappo import train_mappo
from pursuit_lab.replay import ReplayBuffer
from pursuit_lab.rewards import mix_team_rewards
from pursuit_lab.utils import ensure_dir, seed_everything


def checkpoint_payload(
    *,
    policy_net: DQN,
    obs_dim: int,
    action_dim: int,
    config: dict[str, Any],
    episode: int,
    metrics: dict[str, Any],
    epsilon: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": policy_net.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "config": config,
        "episode": episode,
        "metrics": metrics,
        "epsilon": epsilon,
    }


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def eval_score(summary: dict[str, float | int]) -> float:
    return float(summary["capture_rate"]) * 1000.0 - float(summary["mean_steps_to_capture"])


def train_dqn(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    seed_everything(seed)

    run_dir = ensure_dir(Path(config["output_dir"]) / str(config["experiment_name"]))
    save_config(config, run_dir / "config.yaml")

    device = torch.device("cpu")
    env = make_env(seed=seed, env_config=config["env"])
    try:
        obs_dim = int(np.prod(env.observation_space(PURSUER_AGENTS[0]).shape))
        action_dim = int(env.action_space(PURSUER_AGENTS[0]).n)
    finally:
        env.close()

    policy_net = DQN(obs_dim, action_dim, tuple(config["dqn"]["hidden_sizes"])).to(device)
    target_net = DQN(obs_dim, action_dim, tuple(config["dqn"]["hidden_sizes"])).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=float(config["dqn"]["learning_rate"]))
    replay = ReplayBuffer(capacity=int(config["training"]["buffer_capacity"]), seed=seed)

    total_episodes = int(config["total_episodes"])
    max_cycles = int(config["env"].get("max_cycles", 50))
    eval_interval = int(config["evaluation"]["eval_interval"])
    best_score = float("-inf")
    global_step = 0
    metrics_rows: list[dict[str, Any]] = []

    for episode_index in range(total_episodes):
        curriculum_enabled = bool(config["training"].get("curriculum", False))
        stage = curriculum_stage_for_episode(episode_index, total_episodes) if curriculum_enabled else 2
        env = make_env(
            seed=seed + episode_index,
            env_config=config["env"],
            curriculum_stage=stage if curriculum_enabled else None,
        )
        rng = np.random.default_rng(seed + episode_index)
        losses: list[float] = []
        captured = False
        episode_reward = 0.0
        steps = 0
        epsilon = linear_epsilon(
            episode_index,
            epsilon_start=float(config["training"]["epsilon_start"]),
            epsilon_end=float(config["training"]["epsilon_end"]),
            decay_episodes=int(config["training"]["epsilon_decay_episodes"]),
        )

        try:
            observations, _ = env.reset(seed=seed + episode_index)
            if curriculum_enabled:
                apply_curriculum_stage(env, stage)
            for step in range(max_cycles):
                actions: dict[str, int] = {}
                for agent in env.agents:
                    action_space = env.action_space(agent)
                    if agent in PURSUER_AGENTS:
                        actions[agent] = select_epsilon_greedy_action(
                            network=policy_net,
                            observation=observations[agent],
                            epsilon=epsilon,
                            action_dim=action_dim,
                            rng=rng,
                            device=device,
                        )
                    else:
                        actions[agent] = int(action_space.sample())

                next_observations, raw_rewards, terminations, truncations, _ = env.step(actions)
                mixed_rewards = mix_team_rewards(
                    raw_rewards,
                    team_weight=float(config["training"].get("team_reward_weight", 0.0)),
                )
                done_all = all(terminations.values()) or all(truncations.values()) or not env.agents
                captured = captured or any(bool(value) for value in terminations.values())
                episode_reward += sum(float(raw_rewards.get(agent, 0.0)) for agent in PURSUER_AGENTS)
                steps = step + 1

                for agent in PURSUER_AGENTS:
                    if agent not in observations:
                        continue
                    replay.push(
                        observation=observations[agent],
                        action=actions[agent],
                        reward=mixed_rewards[agent],
                        next_observation=next_observations.get(agent, np.zeros(obs_dim, dtype=np.float32)),
                        done=bool(terminations.get(agent, False) or truncations.get(agent, False) or done_all),
                    )

                if len(replay) >= int(config["training"]["min_buffer_size"]):
                    batch = replay.sample(int(config["training"]["batch_size"]))
                    loss = optimize_dqn(
                        policy_net=policy_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        batch=batch,
                        gamma=float(config["dqn"]["gamma"]),
                        device=device,
                    )
                    losses.append(loss)
                    if global_step % int(config["training"]["target_update_interval"]) == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                observations = next_observations
                global_step += 1
                if done_all:
                    break
        finally:
            env.close()

        row: dict[str, Any] = {
            "episode": episode_index + 1,
            "epsilon": epsilon,
            "episode_reward": episode_reward,
            "captured": captured,
            "steps": steps,
            "loss": mean(losses) if losses else 0.0,
            "curriculum_stage": stage,
        }

        should_eval = (episode_index + 1) % eval_interval == 0 or episode_index + 1 == total_episodes
        if should_eval:
            eval_rows, _ = run_eval_episodes(
                policy_net=policy_net,
                env_config=config["env"],
                episodes=int(config["evaluation"]["eval_episodes"]),
                seeds=list(config["evaluation"]["seeds"]),
            )
            summary = summarize_episodes(eval_rows, max_cycles=max_cycles)
            row.update({f"eval_{key}": value for key, value in summary.items()})
            score = eval_score(summary)
            if score >= best_score:
                best_score = score
                save_checkpoint(
                    checkpoint_payload(
                        policy_net=policy_net,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        config=config,
                        episode=episode_index + 1,
                        metrics=summary,
                        epsilon=epsilon,
                    ),
                    run_dir / "best.pt",
                )

        metrics_rows.append(row)
        write_csv(metrics_rows, run_dir / "metrics.csv")

    if not (run_dir / "best.pt").exists():
        save_checkpoint(
            checkpoint_payload(
                policy_net=policy_net,
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config,
                episode=total_episodes,
                metrics=metrics_rows[-1] if metrics_rows else {},
                epsilon=float(config["training"]["epsilon_end"]),
            ),
            run_dir / "best.pt",
        )

    final_rows, _ = run_eval_episodes(
        policy_net=policy_net,
        env_config=config["env"],
        episodes=int(config["evaluation"]["eval_episodes"]),
        seeds=list(config["evaluation"]["seeds"]),
    )
    final_summary = summarize_episodes(final_rows, max_cycles=max_cycles)
    write_csv(final_rows, run_dir / "eval_episodes.csv")
    write_json(final_summary, run_dir / "eval_summary.json")
    return run_dir


def run_random_baseline(config: dict[str, Any]) -> Path:
    run_dir = ensure_dir(Path(config["output_dir"]) / str(config["experiment_name"]))
    save_config(config, run_dir / "config.yaml")
    rows, _ = run_eval_episodes(
        policy_net=None,
        env_config=config["env"],
        episodes=int(config["evaluation"]["eval_episodes"]),
        seeds=list(config["evaluation"]["seeds"]),
    )
    summary = summarize_episodes(rows, max_cycles=int(config["env"].get("max_cycles", 50)))
    write_csv(rows, run_dir / "metrics.csv")
    write_csv(rows, run_dir / "eval_episodes.csv")
    write_json(summary, run_dir / "eval_summary.json")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate pursuit_lab experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--episodes", type=int, default=None, help="Override total_episodes from the config.")
    parser.add_argument("--experiment-name", default=None, help="Override experiment_name from the config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from the config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from the config.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.episodes is not None:
        config["total_episodes"] = args.episodes
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    if config.get("algorithm") == "random" or config["experiment_name"] == "random_baseline":
        run_dir = run_random_baseline(config)
    elif config.get("algorithm") == "mappo":
        run_dir = train_mappo(config)
    else:
        run_dir = train_dqn(config)
    print(run_dir)


if __name__ == "__main__":
    main()
