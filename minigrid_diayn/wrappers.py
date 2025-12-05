"""
MiniGrid environment wrappers for DIAYN.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import minigrid  # Required to register MiniGrid environments


class FullyObsWrapper(gym.ObservationWrapper):
    """
    Use the full grid observation instead of the agent's partial view.

    MiniGrid by default gives the agent a 7x7 partial view.
    This wrapper provides the full grid instead.
    """

    def __init__(self, env):
        super().__init__(env)

        # Get the full grid shape
        self.grid_size = env.unwrapped.width * env.unwrapped.height

        # Update observation space to full grid
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(env.unwrapped.width, env.unwrapped.height, 3),
            dtype=np.uint8
        )

        self.observation_space = spaces.Dict({
            'image': new_image_space,
            'direction': spaces.Discrete(4),
            'mission': spaces.Text(max_length=128)
        })

    def observation(self, obs):
        """Return full grid observation."""
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            10,  # Agent object type
            0,   # Color (unused for agent)
            env.agent_dir  # Agent direction
        ])

        return {
            'image': full_grid,
            'direction': obs['direction'],
            'mission': obs['mission']
        }


class FlatObsWrapper(gym.ObservationWrapper):
    """
    Flatten the grid observation to a 1D vector.

    Takes the 'image' from observation dict and flattens it.
    Also includes the agent direction as part of the observation.
    """

    def __init__(self, env):
        super().__init__(env)

        # Get original image shape
        if isinstance(env.observation_space, spaces.Dict):
            img_space = env.observation_space['image']
        else:
            img_space = env.observation_space

        # Calculate flattened size: image + direction one-hot (4)
        img_size = np.prod(img_space.shape)
        self.obs_size = img_size + 4  # +4 for direction one-hot

        # New observation space is a flat vector
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.obs_size,),
            dtype=np.float32
        )

    def observation(self, obs):
        """Flatten observation to 1D vector."""
        if isinstance(obs, dict):
            image = obs['image'].flatten().astype(np.float32)
            direction = obs['direction']
        else:
            image = obs.flatten().astype(np.float32)
            direction = 0

        # One-hot encode direction
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[direction] = 1.0

        # Normalize image values to [0, 1]
        image = image / 255.0

        # Concatenate
        flat_obs = np.concatenate([image, direction_onehot])

        return flat_obs


class MinigridInfoWrapper(gym.Wrapper):
    """
    Wrapper that adds useful info to the step return.

    Adds agent position and direction to info dict.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add agent position and direction to info
        info['agent_pos'] = self.unwrapped.agent_pos
        info['agent_dir'] = self.unwrapped.agent_dir

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Add agent position and direction to info
        info['agent_pos'] = self.unwrapped.agent_pos
        info['agent_dir'] = self.unwrapped.agent_dir

        return obs, info


def make_env(env_name: str, fully_observable: bool = True, seed: int = None):
    """
    Create a wrapped MiniGrid environment.

    Args:
        env_name: Name of the MiniGrid environment
        fully_observable: If True, use full grid instead of partial view
        seed: Random seed for reproducibility

    Returns:
        Wrapped gymnasium environment
    """
    env = gym.make(env_name)

    # Add info wrapper first
    env = MinigridInfoWrapper(env)

    # Optionally use full observability
    if fully_observable:
        env = FullyObsWrapper(env)

    # Flatten observation
    env = FlatObsWrapper(env)

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    return env


def get_env_info(env):
    """
    Get environment information for network initialization.

    Args:
        env: Wrapped MiniGrid environment

    Returns:
        dict with obs_dim and num_actions
    """
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return {
        'obs_dim': obs_dim,
        'num_actions': num_actions
    }
