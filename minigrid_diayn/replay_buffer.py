"""
Replay buffer for DIAYN training.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.

    Stores transitions: (state, action, reward, next_state, done, skill, disc_next_obs)
    """

    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        skill: int,
        disc_next_obs: np.ndarray = None
    ):
        """
        Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received (pseudo-reward for DIAYN)
            next_state: Next state
            done: Whether episode ended
            skill: Skill index used
            disc_next_obs: Discriminator observation for next state (6-dim)
        """
        self.buffer.append((state, action, reward, next_state, done, skill, disc_next_obs))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones, skills, disc_next_obs)
        """
        # Ensure we don't sample more than available
        batch_size = min(batch_size, len(self.buffer))

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, skills, disc_next_obs = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(skills, dtype=np.int64),
            np.array(disc_next_obs, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()


class SkillBalancedBuffer(ReplayBuffer):
    """
    Replay buffer that maintains separate buffers for each skill.

    This helps ensure balanced sampling across skills.
    """

    def __init__(self, capacity: int = 100000, num_skills: int = 8):
        """
        Args:
            capacity: Total maximum capacity (divided among skills)
            num_skills: Number of skills
        """
        super().__init__(capacity)
        self.num_skills = num_skills
        self.skill_buffers = [
            deque(maxlen=capacity // num_skills) for _ in range(num_skills)
        ]

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        skill: int,
        disc_next_obs: np.ndarray = None
    ):
        """Store transition in both main buffer and skill-specific buffer."""
        super().push(state, action, reward, next_state, done, skill, disc_next_obs)
        self.skill_buffers[skill].append(
            (state, action, reward, next_state, done, skill, disc_next_obs)
        )

    def sample_balanced(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample with balanced representation of skills.

        Args:
            batch_size: Total number of transitions to sample

        Returns:
            Tuple of numpy arrays
        """
        samples_per_skill = batch_size // self.num_skills
        remainder = batch_size % self.num_skills

        batch = []
        for skill_idx in range(self.num_skills):
            skill_buffer = self.skill_buffers[skill_idx]
            if len(skill_buffer) == 0:
                continue

            # Sample from this skill's buffer
            n_samples = samples_per_skill + (1 if skill_idx < remainder else 0)
            n_samples = min(n_samples, len(skill_buffer))
            batch.extend(random.sample(list(skill_buffer), n_samples))

        # If not enough samples, fill from main buffer
        if len(batch) < batch_size:
            needed = batch_size - len(batch)
            if len(self.buffer) >= needed:
                batch.extend(random.sample(list(self.buffer), needed))

        if len(batch) == 0:
            return self.sample(batch_size)

        random.shuffle(batch)
        states, actions, rewards, next_states, dones, skills, disc_next_obs = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(skills, dtype=np.int64),
            np.array(disc_next_obs, dtype=np.float32)
        )
