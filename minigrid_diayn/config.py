"""
Configuration for DIAYN on MiniGrid environments.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class DIAYNConfig:
    """Configuration for DIAYN training on MiniGrid."""

    # Environment (v0 for minigrid 3.0+)
    env_name: str = "MiniGrid-Empty-8x8-v0"

    # Skills
    num_skills: int = 8

    # Training
    num_episodes: int = 2000
    max_steps: int = 100
    batch_size: int = 64
    buffer_size: int = 100000
    start_training_step: int = 1000  # Random exploration before training
    updates_per_step: int = 1

    # Learning rates
    lr_policy: float = 3e-4
    lr_critic: float = 3e-4
    lr_discriminator: float = 3e-4
    lr_alpha: float = 3e-4

    # RL hyperparameters
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient

    # Entropy tuning
    # DIAYN paper uses FIXED alpha = 0.1
    alpha: float = 0.1
    auto_entropy_tuning: bool = False  

    # Network architecture
    hidden_dim: int = 256

    # Discriminator observation dimension
    # 6 = (x, y) normalized + direction one-hot (4)
    disc_obs_dim: int = 6

    # Reward clipping (for stability)
    reward_clip: float = 5.0

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    # Device
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    save_dir: str = "checkpoints"
    plot_dir: str = "plots"

    # Random seed
    seed: int = 42

    def __post_init__(self):
        """Validate and adjust config after initialization."""
        # Ensure device is valid
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"


def get_config(**kwargs) -> DIAYNConfig:
    """Create a config with optional overrides."""
    return DIAYNConfig(**kwargs)
