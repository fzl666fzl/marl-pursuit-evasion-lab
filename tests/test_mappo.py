from __future__ import annotations

import numpy as np
import torch

from pursuit_lab.mappo import MAPPOActor, MAPPOCritic, compute_gae, sample_actor_action


def test_mappo_actor_and_critic_dimensions() -> None:
    actor = MAPPOActor(obs_dim=16, action_dim=5, hidden_sizes=(32,))
    critic = MAPPOCritic(global_obs_dim=62, hidden_sizes=(32,))

    logits = actor(torch.zeros((4, 16), dtype=torch.float32))
    values = critic(torch.zeros((4, 62), dtype=torch.float32))

    assert logits.shape == (4, 5)
    assert values.shape == (4,)


def test_mappo_action_sampling_returns_valid_discrete_action() -> None:
    actor = MAPPOActor(obs_dim=16, action_dim=5, hidden_sizes=(32,))
    observation = np.zeros(16, dtype=np.float32)

    action, log_prob = sample_actor_action(actor, observation)

    assert 0 <= action < 5
    assert isinstance(log_prob, float)


def test_compute_gae_matches_discounted_returns_when_lambda_is_one() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.9,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, np.array([2.71, 1.9, 1.0], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(returns, advantages, rtol=1e-5)
