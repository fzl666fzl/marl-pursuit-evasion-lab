from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from pursuit_lab.replay import TransitionBatch


class DQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden_size)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_size)
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def select_epsilon_greedy_action(
    *,
    network: DQN,
    observation: np.ndarray,
    epsilon: float,
    action_dim: int,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(action_dim))
    with torch.no_grad():
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, -1)
        return int(network(tensor).argmax(dim=1).item())


def linear_epsilon(
    episode_index: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    decay_episodes: int,
) -> float:
    if decay_episodes <= 0:
        return epsilon_end
    fraction = min(1.0, episode_index / decay_episodes)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


def optimize_dqn(
    *,
    policy_net: DQN,
    target_net: DQN,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    gamma: float,
    device: torch.device | str = "cpu",
) -> float:
    observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)

    q_values = policy_net(observations).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_observations).max(dim=1).values
        targets = rewards + gamma * next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())
