"""Configuration dataclasses for DIAYN training."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from .utils import get_device


@dataclass
class DIAYNConfig:
    """Configuration for DIAYN skill discovery training."""

    # Environment
    env_key: str = "empty-8x8"

    # Skills
    num_skills: int = 8

    # Training
    num_episodes: int = 10000
    max_steps: int = 10  # Steps per episode
    batch_size: int = 256
    buffer_size: int = 10000
    start_training_step: int = 1000  # Begin training after this many steps

    # Learning rates
    lr_policy: float = 1e-4
    lr_discriminator: float = 1e-4

    # Entropy coefficient for exploration
    entropy_coef: float = 0.01

    # Number of gradient updates per episode
    updates_per_episode: int = 1

    # Network architecture
    hidden_dim: int = 256
    feature_dim: int = 64  # CNN encoder output dimension

    # Environment settings
    random_start: bool = False  # Randomize agent starting position
    partial_obs: bool = False  # Use partial 7x7 view (True) or full grid (False)

    # Discriminator type: "state" uses encoded features, "position" uses (x,y) coordinates
    discriminator_type: str = "state"

    # Freeze discriminator for first N episodes (allows policy to explore first)
    freeze_disc_episodes: int = 0

    # Restrict to movement actions only {left, right, forward}
    # Prevents "camping" equilibrium from no-op actions
    movement_only: bool = False

    # Logging
    log_interval: int = 10
    save_interval: int = 500

    # Device and seed
    device: str = field(default_factory=get_device)
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DIAYNConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical controller training."""

    num_skills: int = 8  # Number of pre-trained skills to select from
    skill_duration: int = 8  # Steps to execute each selected skill
    meta_hidden_dim: int = 128  # Meta-controller hidden layer size
    meta_lr: float = 3e-4  # Meta-controller learning rate
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft target update rate
    batch_size: int = 64  # Training batch size
    buffer_size: int = 50000  # Replay buffer capacity
    goal_dim: int = 2  # Goal vector dimension (x, y target position)
    meta_obs_dim: int = 6  # Meta-controller input: position (2) + direction (4)
    entropy_coef: float = 0.1  # Entropy bonus for exploration
    device: str = field(default_factory=get_device)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HierarchicalConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
