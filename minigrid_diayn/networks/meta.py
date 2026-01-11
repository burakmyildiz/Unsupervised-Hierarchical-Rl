"""Meta-controller networks for hierarchical DIAYN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init_weights(m, gain=1.0):
    """Initialize network weights with orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class MetaController(nn.Module):
    """High-level policy that selects skills given state and goal."""

    def __init__(self, obs_dim: int, goal_dim: int, hidden_dim: int, num_skills: int):
        super().__init__()
        self.num_skills = num_skills

        self.network = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))
        init_weights(self.network[-1], gain=0.01)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.network(x)

    def get_skill_probs(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(obs, goal), dim=-1)

    def sample(self, obs: torch.Tensor, goal: torch.Tensor):
        """Sample skill and return log_prob."""
        logits = self.forward(obs, goal)
        dist = Categorical(logits=logits)
        skill = dist.sample()
        log_prob = dist.log_prob(skill)
        return skill, log_prob

    def select_skill(self, obs: torch.Tensor, goal: torch.Tensor, deterministic: bool = False):
        """Select skill for environment interaction."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            goal = goal.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        with torch.no_grad():
            logits = self.forward(obs, goal)
            if deterministic:
                skill = torch.argmax(logits, dim=-1)
            else:
                skill = Categorical(logits=logits).sample()

        return skill.item() if squeeze else skill


class MetaQNetwork(nn.Module):
    """Q-network for meta-controller: Q(s, g, z)."""

    def __init__(self, obs_dim: int, goal_dim: int, hidden_dim: int, num_skills: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.network(x)
