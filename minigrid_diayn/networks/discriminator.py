"""Discriminator networks for DIAYN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(m, gain=1.0):
    """Initialize network weights with orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class StateDiscriminator(nn.Module):
    """
    State-based discriminator for DIAYN (matches reference implementation).

    Predicts skill from ENCODED STATE features, not position.
    This is how the reference Skill-Discovery-Agent works.

    Input: Encoded state features from encoder (e.g., 64-dim or 128-dim)
    Output: Skill logits (num_skills)
    """

    def __init__(self, input_dim: int, num_skills: int, hidden_dim: int = 256):
        """
        Args:
            input_dim: Dimension of encoded state features (from encoder)
            num_skills: Number of skills to classify
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_skills = num_skills

        # MLP: encoded_state -> hidden -> hidden -> skill_logits
        # Matches reference: 2 hidden layers with ReLU
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )

        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))

    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get skill logits.

        Args:
            encoded_state: Encoded state features (batch, input_dim)

        Returns:
            skill_logits: Logits for each skill (batch, num_skills)
        """
        return self.network(encoded_state)

    def get_log_probs(self, encoded_state: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for all skills."""
        logits = self.forward(encoded_state)
        return F.log_softmax(logits, dim=-1)

    def compute_reward(self, encoded_state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Compute intrinsic reward (matches reference implementation).

        Args:
            encoded_state: Encoded state features (batch, input_dim)
            skill: One-hot skill vectors (batch, num_skills)

        Returns:
            reward: Intrinsic reward = log q(z|s) (batch,)
                    Note: Reference does NOT subtract log p(z)!
        """
        with torch.no_grad():
            logits = self.forward(encoded_state)
            log_probs = F.log_softmax(logits, dim=-1)
            # Dot product with skill one-hot to get log q(z|s)
            reward = (log_probs * skill).sum(dim=-1)
        return reward


class PositionDiscriminator(nn.Module):
    """
    Position-based discriminator for DIAYN.

    Predicts skill from (x, y) position only, not full state.
    This is what the original DIAYN paper recommends - forces skills
    to differentiate by WHERE they go, not by memorizing full states.

    CRITICAL: Uses geometric initialization to bootstrap learning.
    Without this, all skills start with identical random walk distributions,
    and the discriminator outputs uniform (no learning signal).
    """

    def __init__(self, hidden_dim: int, num_skills: int, geometric_init: bool = True):
        """
        Args:
            hidden_dim: Hidden layer dimension
            num_skills: Number of skills to classify
            geometric_init: If True, initialize with sector-based geometric prior
        """
        super().__init__()

        self.num_skills = num_skills
        self.geometric_init = geometric_init

        # MLP: position (2) -> hidden -> hidden -> skill_logits
        # Smaller network since input is only 2D
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )

        if geometric_init:
            self._init_geometric()
        else:
            # Standard orthogonal initialization
            self.apply(lambda m: init_weights(m, gain=math.sqrt(2)))

    def _init_geometric(self):
        """Initialize discriminator with sector-based geometric prior.

        Creates a simple linear function that divides the position space
        into regions. For 8 skills, uses cardinal and diagonal directions:
        - Skill 0: Top (y > 0.5, |x-0.5| < |y-0.5|)
        - Skill 1: Top-Right (x > 0.5, y > 0.5, x-0.5 > y-0.5)
        - Skill 2: Right
        - etc.

        This bootstraps learning by giving non-uniform predictions from step 1.
        """
        # Initialize first layers to pass through position features
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)

        # Override first layer to create useful features from (x, y)
        first_layer = self.network[0]
        with torch.no_grad():
            # Create direction-sensitive features
            # Input is (x, y) normalized to [0, 1]
            # We want features that respond to different directions from center (0.5, 0.5)

            # Zero out and set specific patterns
            first_layer.weight.zero_()
            first_layer.bias.zero_()

            hidden_dim = first_layer.weight.shape[0]

            # Feature 0-7: respond to 8 directions (if hidden_dim >= 8)
            # Each feature has high response in one sector
            for i in range(min(8, hidden_dim)):
                angle = 2 * np.pi * i / 8  # 8 sectors
                # Direction vector from center
                dx, dy = np.cos(angle), np.sin(angle)
                # Weight: dot product with direction
                first_layer.weight[i, 0] = dx * 3.0  # x weight
                first_layer.weight[i, 1] = dy * 3.0  # y weight
                # Bias to center at (0.5, 0.5)
                first_layer.bias[i] = -(dx * 0.5 + dy * 0.5) * 3.0

            # Remaining features: random (will be learned)
            for i in range(8, hidden_dim):
                first_layer.weight[i, 0] = np.random.randn() * 0.5
                first_layer.weight[i, 1] = np.random.randn() * 0.5
                first_layer.bias[i] = np.random.randn() * 0.5

        # Override final layer to map directional features to skills
        final_layer = self.network[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.zero_()

            # Map direction feature i to skill i
            for i in range(min(self.num_skills, 8)):
                if i < final_layer.weight.shape[1]:  # Check hidden_dim
                    final_layer.weight[i, i] = 2.0  # Strong connection

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get skill logits.

        Args:
            position: Normalized position (batch, 2) in range [0, 1]

        Returns:
            skill_logits: Logits for each skill (batch, num_skills)
        """
        return self.network(position)

    def get_log_probs(self, position: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for all skills."""
        logits = self.forward(position)
        return F.log_softmax(logits, dim=-1)
