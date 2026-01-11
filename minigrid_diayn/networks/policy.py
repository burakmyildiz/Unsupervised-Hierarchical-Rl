"""Discrete policy network for DIAYN."""

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_weights(m, gain=1.0):
    """Initialize network weights with orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class DiscretePolicy(nn.Module):
    """
    Categorical policy network for discrete action spaces.

    Takes observation and skill one-hot as input.
    Outputs action probabilities.
    """

    def __init__(self, obs_dim: int, skill_dim: int, hidden_dim: int, num_actions: int):
        """
        Args:
            obs_dim: Dimension of observation space
            skill_dim: Number of skills (for one-hot encoding)
            hidden_dim: Hidden layer dimension
            num_actions: Number of discrete actions
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.num_actions = num_actions

        # MLP: [obs + skill_onehot] -> hidden -> 2*hidden -> hidden -> action_logits
        # LeakyReLU + Dropout prevent entropy collapse during training
        self.network = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )

        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
        # Smaller initialization for output layer
        init_weights(self.network[-1], gain=0.01)

    def forward(self, obs: torch.Tensor, skill_onehot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get action logits.

        Args:
            obs: Observation tensor (batch, obs_dim)
            skill_onehot: Skill one-hot tensor (batch, skill_dim)

        Returns:
            action_logits: Logits for each action (batch, num_actions)
        """
        x = torch.cat([obs, skill_onehot], dim=-1)
        return self.network(x)

    def get_action(self, obs: torch.Tensor, skill_onehot: torch.Tensor, deterministic: bool = False):
        """
        Get action for environment interaction.

        Args:
            obs: Observation tensor (batch, obs_dim) or (obs_dim,)
            skill_onehot: Skill one-hot tensor (batch, skill_dim) or (skill_dim,)
            deterministic: If True, return argmax action

        Returns:
            action: Action to take (int or tensor)
        """
        # Add batch dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            skill_onehot = skill_onehot.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        with torch.no_grad():
            logits = self.forward(obs, skill_onehot)

            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()

        if squeeze:
            return action.item()
        return action
