"""Replay buffer for DIAYN training."""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Experience replay buffer storing (state, action, reward, next_state, done, skill, position)."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, skill: int,
             position: np.ndarray = None):
        """Push transition with optional position (for discriminator)."""
        self.buffer.append((state, action, reward, next_state, done, skill, position))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, skills, positions = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(skills, dtype=np.int64),
            np.array(positions, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def clear(self):
        self.buffer.clear()
