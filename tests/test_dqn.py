from __future__ import annotations

import numpy as np
import torch

from pursuit_lab.dqn import DQN, select_epsilon_greedy_action
from pursuit_lab.envs import make_env


def test_dqn_dimensions_match_adversary_spaces() -> None:
    env = make_env(seed=0)
    try:
        obs_dim = int(np.prod(env.observation_space("adversary_0").shape))
        action_dim = int(env.action_space("adversary_0").n)
        network = DQN(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(32,))

        output = network(torch.zeros((4, obs_dim), dtype=torch.float32))

        assert output.shape == (4, action_dim)
    finally:
        env.close()


def test_epsilon_greedy_policy_returns_valid_discrete_action() -> None:
    network = DQN(obs_dim=3, action_dim=5, hidden_sizes=(8,))
    observation = np.zeros(3, dtype=np.float32)

    action = select_epsilon_greedy_action(
        network=network,
        observation=observation,
        epsilon=0.0,
        action_dim=5,
        rng=np.random.default_rng(0),
    )

    assert 0 <= action < 5
