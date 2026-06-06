from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn

from pursuit_lab.constants import PREY_AGENT, PURSUER_AGENTS
from pursuit_lab.envs import make_env
from pursuit_lab.evaluation import run_eval_episodes
from pursuit_lab.metrics import append_csv_row, summarize_episodes, write_csv, write_json
from pursuit_lab.utils import ensure_dir, seed_everything

MAPPO_METRIC_FIELDS = [
    "episode",
    "episode_reward",
    "captured",
    "steps",
    "loss",
    "curriculum_stage",
    "eval_episodes",
    "eval_capture_rate",
    "eval_mean_episode_reward",
    "eval_mean_steps_to_capture",
    "eval_success_episode_count",
]


class MAPPOActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class MAPPOCritic(nn.Module):
    def __init__(self, global_obs_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        self.net = build_mlp(global_obs_dim, hidden_sizes, 1)

    def forward(self, global_observations: torch.Tensor) -> torch.Tensor:
        return self.net(global_observations).squeeze(-1)


@dataclass(frozen=True)
class MAPPOTransition:
    agent: str
    observation: np.ndarray
    global_observation: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


def build_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_size)))
        layers.append(nn.Tanh())
        last_dim = int(hidden_size)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def sample_actor_action(
    actor: MAPPOActor,
    observation: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> tuple[int, float]:
    with torch.no_grad():
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
        distribution = torch.distributions.Categorical(logits=actor(tensor))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
    return int(action.item()), float(log_prob.item())


def build_global_observation(
    observations: dict[str, np.ndarray],
    *,
    pursuer_obs_dim: int,
    prey_obs_dim: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for agent in PURSUER_AGENTS:
        if agent in observations:
            parts.append(np.asarray(observations[agent], dtype=np.float32).reshape(-1))
        else:
            parts.append(np.zeros(pursuer_obs_dim, dtype=np.float32))
    if PREY_AGENT in observations:
        parts.append(np.asarray(observations[PREY_AGENT], dtype=np.float32).reshape(-1))
    else:
        parts.append(np.zeros(prey_obs_dim, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    next_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        following_value = next_value if index == len(rewards) - 1 else float(values[index + 1])
        nonterminal = 1.0 - float(dones[index])
        delta = float(rewards[index]) + gamma * following_value * nonterminal - float(values[index])
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values.astype(np.float32)
    return advantages.astype(np.float32), returns.astype(np.float32)


def optimize_mappo(
    *,
    actor: MAPPOActor,
    critic: MAPPOCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[MAPPOTransition],
    config: dict[str, Any],
    device: torch.device | str = "cpu",
) -> float:
    if not transitions:
        return 0.0

    advantages = np.zeros(len(transitions), dtype=np.float32)
    returns = np.zeros(len(transitions), dtype=np.float32)
    for agent in PURSUER_AGENTS:
        indexes = [idx for idx, item in enumerate(transitions) if item.agent == agent]
        if not indexes:
            continue
        rewards = np.asarray([transitions[idx].reward for idx in indexes], dtype=np.float32)
        values = np.asarray([transitions[idx].value for idx in indexes], dtype=np.float32)
        dones = np.asarray([transitions[idx].done for idx in indexes], dtype=np.float32)
        agent_advantages, agent_returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
        )
        advantages[indexes] = agent_advantages
        returns[indexes] = agent_returns

    observations = torch.as_tensor(
        np.stack([item.observation for item in transitions]).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    global_observations = torch.as_tensor(
        np.stack([item.global_observation for item in transitions]).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor([item.action for item in transitions], dtype=torch.int64, device=device)
    old_log_probs = torch.as_tensor([item.log_prob for item in transitions], dtype=torch.float32, device=device)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if advantage_tensor.numel() > 1:
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std(unbiased=False) + 1e-8)

    losses: list[float] = []
    for _ in range(int(config["update_epochs"])):
        distribution = torch.distributions.Categorical(logits=actor(observations))
        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        ratio = torch.exp(new_log_probs - old_log_probs)

        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(
            ratio,
            1.0 - float(config["clip_coef"]),
            1.0 + float(config["clip_coef"]),
        ) * advantage_tensor
        policy_loss = -torch.min(unclipped, clipped).mean()
        values = critic(global_observations)
        value_loss = nn.functional.mse_loss(values, return_tensor)
        loss = policy_loss + float(config["value_coef"]) * value_loss - float(config["entropy_coef"]) * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            max_norm=float(config["max_grad_norm"]),
        )
        optimizer.step()
        losses.append(float(loss.item()))

    return mean(losses) if losses else 0.0


def mappo_checkpoint_payload(
    *,
    actor: MAPPOActor,
    critic: MAPPOCritic,
    obs_dim: int,
    prey_obs_dim: int,
    global_obs_dim: int,
    action_dim: int,
    config: dict[str, Any],
    episode: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "algorithm": "mappo",
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "obs_dim": obs_dim,
        "prey_obs_dim": prey_obs_dim,
        "global_obs_dim": global_obs_dim,
        "action_dim": action_dim,
        "config": config,
        "episode": episode,
        "metrics": metrics,
    }


def save_mappo_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def eval_score(summary: dict[str, float | int]) -> float:
    return float(summary["capture_rate"]) * 1000.0 - float(summary["mean_steps_to_capture"])


def train_mappo(config: dict[str, Any]) -> Path:
    seed = int(config["seed"])
    seed_everything(seed)

    run_dir = ensure_dir(Path(config["output_dir"]) / str(config["experiment_name"]))
    from pursuit_lab.config import save_config

    save_config(config, run_dir / "config.yaml")

    device = torch.device("cpu")
    env = make_env(seed=seed, env_config=config["env"])
    try:
        obs_dim = int(np.prod(env.observation_space(PURSUER_AGENTS[0]).shape))
        prey_obs_dim = int(np.prod(env.observation_space(PREY_AGENT).shape))
        action_dim = int(env.action_space(PURSUER_AGENTS[0]).n)
    finally:
        env.close()

    global_obs_dim = obs_dim * len(PURSUER_AGENTS) + prey_obs_dim
    mappo_config = dict(config["mappo"])
    actor = MAPPOActor(obs_dim, action_dim, tuple(mappo_config["actor_hidden_sizes"])).to(device)
    critic = MAPPOCritic(global_obs_dim, tuple(mappo_config["critic_hidden_sizes"])).to(device)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=float(mappo_config["learning_rate"]),
    )

    total_episodes = int(config["total_episodes"])
    max_cycles = int(config["env"].get("max_cycles", 50))
    eval_interval = int(config["evaluation"]["eval_interval"])
    rollout_episodes = int(mappo_config["rollout_episodes"])
    best_score = float("-inf")
    metrics_rows: list[dict[str, Any]] = []
    metrics_path = run_dir / "metrics.csv"
    metrics_path.write_text("", encoding="utf-8")
    rollout: list[MAPPOTransition] = []

    for episode_index in range(total_episodes):
        env = make_env(seed=seed + episode_index, env_config=config["env"])
        captured = False
        episode_reward = 0.0
        steps = 0

        try:
            observations, _ = env.reset(seed=seed + episode_index)
            for step in range(max_cycles):
                global_observation = build_global_observation(
                    observations,
                    pursuer_obs_dim=obs_dim,
                    prey_obs_dim=prey_obs_dim,
                )
                global_tensor = torch.as_tensor(global_observation, dtype=torch.float32, device=device).reshape(1, -1)
                with torch.no_grad():
                    state_value = float(critic(global_tensor).item())

                actions: dict[str, int] = {}
                pending: list[tuple[str, np.ndarray, np.ndarray, int, float, float]] = []
                for agent in env.agents:
                    action_space = env.action_space(agent)
                    if agent in PURSUER_AGENTS:
                        action, log_prob = sample_actor_action(actor, observations[agent], device=device)
                        actions[agent] = action
                        pending.append(
                            (
                                agent,
                                np.asarray(observations[agent], dtype=np.float32).reshape(-1),
                                global_observation,
                                action,
                                log_prob,
                                state_value,
                            )
                        )
                    else:
                        actions[agent] = int(action_space.sample())

                next_observations, rewards, terminations, truncations, _ = env.step(actions)
                done_all = all(terminations.values()) or all(truncations.values()) or not env.agents
                captured = captured or any(bool(value) for value in terminations.values())
                episode_reward += sum(float(rewards.get(agent, 0.0)) for agent in PURSUER_AGENTS)
                steps = step + 1

                for agent, observation, state, action, log_prob, value in pending:
                    rollout.append(
                        MAPPOTransition(
                            agent=agent,
                            observation=observation,
                            global_observation=state,
                            action=action,
                            log_prob=log_prob,
                            value=value,
                            reward=float(rewards.get(agent, 0.0)),
                            done=bool(terminations.get(agent, False) or truncations.get(agent, False) or done_all),
                        )
                    )

                observations = next_observations
                if done_all:
                    break
        finally:
            env.close()

        loss = 0.0
        should_update = (episode_index + 1) % rollout_episodes == 0 or episode_index + 1 == total_episodes
        if should_update:
            loss = optimize_mappo(
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                transitions=rollout,
                config=mappo_config,
                device=device,
            )
            rollout.clear()

        row: dict[str, Any] = {
            "episode": episode_index + 1,
            "episode_reward": episode_reward,
            "captured": captured,
            "steps": steps,
            "loss": loss,
            "curriculum_stage": 2,
        }

        should_eval = (episode_index + 1) % eval_interval == 0 or episode_index + 1 == total_episodes
        if should_eval:
            eval_rows, _ = run_eval_episodes(
                policy_net=actor,
                env_config=config["env"],
                episodes=int(config["evaluation"]["eval_episodes"]),
                seeds=list(config["evaluation"]["seeds"]),
            )
            summary = summarize_episodes(eval_rows, max_cycles=max_cycles)
            row.update({f"eval_{key}": value for key, value in summary.items()})
            score = eval_score(summary)
            if score >= best_score:
                best_score = score
                save_mappo_checkpoint(
                    mappo_checkpoint_payload(
                        actor=actor,
                        critic=critic,
                        obs_dim=obs_dim,
                        prey_obs_dim=prey_obs_dim,
                        global_obs_dim=global_obs_dim,
                        action_dim=action_dim,
                        config=config,
                        episode=episode_index + 1,
                        metrics=summary,
                    ),
                    run_dir / "best.pt",
                )

        metrics_rows.append(row)
        append_csv_row(row, metrics_path, fields=MAPPO_METRIC_FIELDS)

    if not (run_dir / "best.pt").exists():
        save_mappo_checkpoint(
            mappo_checkpoint_payload(
                actor=actor,
                critic=critic,
                obs_dim=obs_dim,
                prey_obs_dim=prey_obs_dim,
                global_obs_dim=global_obs_dim,
                action_dim=action_dim,
                config=config,
                episode=total_episodes,
                metrics=metrics_rows[-1] if metrics_rows else {},
            ),
            run_dir / "best.pt",
        )

    final_rows, _ = run_eval_episodes(
        policy_net=actor,
        env_config=config["env"],
        episodes=int(config["evaluation"]["eval_episodes"]),
        seeds=list(config["evaluation"]["seeds"]),
    )
    final_summary = summarize_episodes(final_rows, max_cycles=max_cycles)
    write_csv(final_rows, run_dir / "eval_episodes.csv")
    write_json(final_summary, run_dir / "eval_summary.json")
    return run_dir
