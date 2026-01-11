"""CNN encoders for MiniGrid observations."""

import math

import torch
import torch.nn as nn


def init_weights(m, gain=1.0):
    """Initialize network weights with orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class PartialGridEncoder(nn.Module):
    """CNN encoder for partial 7x7 MiniGrid observations.

    Processes the agent's local view (7x7x3 = 147 dims) and outputs
    a compact feature vector for the policy and discriminator.
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.view_size = 7
        self.feature_dim = feature_dim

        # CNN for spatial features (stride=1 preserves spatial resolution)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate conv output size: 7x7x32 = 1568
        conv_out_size = 7 * 7 * 32

        # FC layer to feature_dim
        self.fc = nn.Linear(conv_out_size, feature_dim)

        # Initialize weights
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        init_weights(self.fc, gain=math.sqrt(2))

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            flat_obs: Flattened observation (batch, 147) - 7x7x3

        Returns:
            features: Encoded features (batch, feature_dim), always positive (ReLU)
        """
        batch_size = flat_obs.shape[0]

        # MiniGrid uses symbolic encoding (0-10), NOT RGB (0-255)
        # Do NOT normalize - values are already small and meaningful

        # Reshape to (batch, H, W, C) then to (batch, C, H, W)
        grid = flat_obs.view(batch_size, self.view_size, self.view_size, 3)
        grid = grid.permute(0, 3, 1, 2).contiguous()

        # CNN forward
        conv_features = self.conv(grid)

        # ReLU ensures non-negative features
        return torch.relu(self.fc(conv_features))


class GridEncoder(nn.Module):
    """CNN encoder for full MiniGrid grid observations.

    Processes flat observation (grid_size² * 3 + 4) and outputs feature vector.
    The last 4 dimensions (direction one-hot) are concatenated after CNN processing.

    Architecture:
    - Stride-1 convolutions preserve spatial resolution
    - ReLU activation ensures non-negative output features
    """

    def __init__(self, grid_size: int, feature_dim: int = 64):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        # CNN for spatial features (stride=1 preserves resolution)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate conv output size dynamically
        conv_out_size = self._get_conv_output_size()

        # FC layer to feature_dim (direction handled via concatenation before policy)
        self.fc = nn.Linear(conv_out_size + 4, feature_dim)

        # Initialize weights
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        init_weights(self.fc, gain=math.sqrt(2))

    def _get_conv_output_size(self) -> int:
        """Calculate CNN output size for given grid_size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.grid_size, self.grid_size)
            out = self.conv(dummy)
            return out.shape[1]

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            flat_obs: Flattened observation (batch, grid_size² * 3 + 4)
                      Values should be normalized by 255

        Returns:
            features: Encoded features (batch, feature_dim), always positive (ReLU)
        """
        batch_size = flat_obs.shape[0]

        # Split grid and direction
        grid_flat = flat_obs[:, :-4]
        direction = flat_obs[:, -4:]

        # MiniGrid uses symbolic encoding (0-10), NOT RGB (0-255)
        # Do NOT normalize - values are already small and meaningful

        # Reshape grid to (batch, H, W, C) then to (batch, C, H, W)
        grid = grid_flat.view(batch_size, self.grid_size, self.grid_size, 3)
        grid = grid.permute(0, 3, 1, 2).contiguous()

        # CNN forward
        conv_features = self.conv(grid)

        # Concatenate with direction and apply FC
        combined = torch.cat([conv_features, direction], dim=-1)

        # ReLU ensures non-negative features
        return torch.relu(self.fc(combined))
