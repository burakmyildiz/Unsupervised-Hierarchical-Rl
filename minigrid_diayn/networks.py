"""
Neural network architectures for DIAYN on MiniGrid.

Includes:
- DiscretePolicy: Categorical policy for discrete actions
- QNetwork: State-action-skill value function
- Discriminator: Skill classifier from states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math


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

        # MLP: [obs + skill_onehot] -> hidden -> hidden -> action_logits
        self.network = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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

    def get_action_probs(self, obs: torch.Tensor, skill_onehot: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax of logits)."""
        logits = self.forward(obs, skill_onehot)
        return F.softmax(logits, dim=-1)

    def sample(self, obs: torch.Tensor, skill_onehot: torch.Tensor):
        """
        Sample action from the policy.

        Args:
            obs: Observation tensor (batch, obs_dim)
            skill_onehot: Skill one-hot tensor (batch, skill_dim)

        Returns:
            action: Sampled action (batch,)
            log_prob: Log probability of the action (batch, 1)
            entropy: Entropy of the distribution (batch, 1)
        """
        logits = self.forward(obs, skill_onehot)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, skill_onehot: torch.Tensor, action: torch.Tensor):
        """
        Evaluate actions for policy gradient.

        Args:
            obs: Observation tensor (batch, obs_dim)
            skill_onehot: Skill one-hot tensor (batch, skill_dim)
            action: Actions to evaluate (batch,)

        Returns:
            log_prob: Log probability of actions (batch, 1)
            entropy: Entropy of the distribution (batch, 1)
        """
        logits = self.forward(obs, skill_onehot)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        return log_prob, entropy

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


class QNetwork(nn.Module):
    """
    Q-Network for discrete actions.

    Takes observation and skill one-hot as input.
    Outputs Q-values for each action.
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

        # MLP: [obs + skill_onehot] -> hidden -> hidden -> Q(s,a,z) for each action
        self.network = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))

    def forward(self, obs: torch.Tensor, skill_onehot: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get Q-values for all actions.

        Args:
            obs: Observation tensor (batch, obs_dim)
            skill_onehot: Skill one-hot tensor (batch, skill_dim)

        Returns:
            q_values: Q-values for each action (batch, num_actions)
        """
        x = torch.cat([obs, skill_onehot], dim=-1)
        return self.network(x)

    def get_q_value(self, obs: torch.Tensor, skill_onehot: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q-value for specific actions.

        Args:
            obs: Observation tensor (batch, obs_dim)
            skill_onehot: Skill one-hot tensor (batch, skill_dim)
            action: Actions (batch,)

        Returns:
            q_value: Q-value for the actions (batch, 1)
        """
        q_values = self.forward(obs, skill_onehot)
        q_value = q_values.gather(1, action.unsqueeze(-1))
        return q_value


class Discriminator(nn.Module):
    """
    Discriminator network for DIAYN.

    Predicts which skill produced a given state.
    This is the q(z|s) network in the DIAYN paper.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, num_skills: int):
        """
        Args:
            obs_dim: Dimension of observation space
            hidden_dim: Hidden layer dimension
            num_skills: Number of skills to classify
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.num_skills = num_skills

        # MLP: obs -> hidden -> hidden -> skill_logits
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )

        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get skill logits.

        Args:
            obs: Observation tensor (batch, obs_dim)

        Returns:
            skill_logits: Logits for each skill (batch, num_skills)
        """
        return self.network(obs)

    def get_log_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for all skills.

        Args:
            obs: Observation tensor (batch, obs_dim)

        Returns:
            log_probs: Log probabilities (batch, num_skills)
        """
        logits = self.forward(obs)
        return F.log_softmax(logits, dim=-1)

    def predict_skill(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely skill.

        Args:
            obs: Observation tensor (batch, obs_dim)

        Returns:
            predicted_skill: Most likely skill (batch,)
        """
        logits = self.forward(obs)
        return torch.argmax(logits, dim=-1)
