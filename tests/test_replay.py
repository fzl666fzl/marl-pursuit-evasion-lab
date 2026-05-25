from __future__ import annotations

import numpy as np

from pursuit_lab.replay import ReplayBuffer


def test_replay_buffer_stores_and_samples_transitions() -> None:
    buffer = ReplayBuffer(capacity=3, seed=0)

    for idx in range(4):
        observation = np.full(2, idx, dtype=np.float32)
        buffer.push(
            observation=observation,
            action=idx % 5,
            reward=float(idx),
            next_observation=observation + 1,
            done=idx % 2 == 0,
        )

    batch = buffer.sample(batch_size=2)

    assert len(buffer) == 3
    assert batch.observations.shape == (2, 2)
    assert batch.actions.shape == (2,)
    assert batch.rewards.shape == (2,)
    assert batch.next_observations.shape == (2, 2)
    assert batch.dones.shape == (2,)
