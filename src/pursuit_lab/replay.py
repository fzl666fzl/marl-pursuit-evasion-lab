from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        *,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            Transition(
                observation=np.asarray(observation, dtype=np.float32).reshape(-1),
                action=int(action),
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32).reshape(-1),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        transitions = self._rng.sample(list(self._buffer), batch_size)
        return TransitionBatch(
            observations=np.stack([item.observation for item in transitions]).astype(np.float32),
            actions=np.asarray([item.action for item in transitions], dtype=np.int64),
            rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
            next_observations=np.stack([item.next_observation for item in transitions]).astype(np.float32),
            dones=np.asarray([item.done for item in transitions], dtype=np.float32),
        )
