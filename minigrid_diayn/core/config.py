"""DIAYN configuration."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from .utils import get_device


@dataclass
class DIAYNConfig:
    # Environment
    env_key: str = "empty-8x8"

    # Skills
    num_skills: int = 8

    # Training (matches reference Skill-Discovery-Agent)
    num_episodes: int = 10000  # Reference uses 10000
    max_steps: int = 10  # Reference uses 10 steps per episode (critical!)
    batch_size: int = 256
    buffer_size: int = 10000  # Reference uses 10000
    start_training_step: int = 1000

    # Learning rates
    lr_policy: float = 1e-4
    lr_discriminator: float = 1e-4

    # Entropy coefficient (reference uses 0.01)
    entropy_coef: float = 0.01

    # Number of training updates per episode
    updates_per_episode: int = 1  # Reference does 1 update per episode

    # Network architecture (match reference)
    hidden_dim: int = 256  # Reference uses 256
    feature_dim: int = 64  # Reference uses 64

    # Environment settings
    random_start: bool = False  # Reference doesn't use random start
    partial_obs: bool = False  # Use full grid observation

    # Discriminator type: "state" (reference) or "position" (geometric init for bootstrap)
    discriminator_type: str = "state"

    # Freeze discriminator for first N episodes (allows policy to adapt to initial bias)
    freeze_disc_episodes: int = 0

    # Restrict actions to movement only {left, right, forward}
    # Removes no-op actions that cause "camping" equilibrium
    movement_only: bool = False

    # Logging
    log_interval: int = 10
    save_interval: int = 500

    # Device and seed
    device: str = field(default_factory=get_device)
    seed: int = 42

    def apply_reference_mode(self):
        """Apply reference implementation settings (defaults already match reference)."""
        pass  # Defaults already match reference implementation

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DIAYNConfig":
        with open(path) as f:
            data = json.load(f)
        # Backward compatibility: remove deprecated params
        for old_param in ["disc_obs_dim", "lr_critic", "updates_per_step", "tau",
                          "normalize_rewards", "diversity_coef", "gamma",
                          "reward_scale", "reference_mode"]:
            data.pop(old_param, None)
        return cls(**data)


@dataclass
class HierarchicalConfig:
    num_skills: int = 8
    skill_duration: int = 8
    meta_hidden_dim: int = 128
    meta_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 50000
    goal_dim: int = 2
    meta_obs_dim: int = 6
    entropy_coef: float = 0.1
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
