"""Multi-environment support for DIAYN experiments."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple
import minigrid

ENVIRONMENTS = {
    "empty-6x6": {"env_name": "MiniGrid-Empty-6x6-v0", "grid_size": 6},
    "empty-8x8": {"env_name": "MiniGrid-Empty-8x8-v0", "grid_size": 8},
    "empty-16x16": {"env_name": "MiniGrid-Empty-16x16-v0", "grid_size": 16},
    "fourrooms": {"env_name": "MiniGrid-FourRooms-v0", "grid_size": 19},
    "doorkey-5x5": {"env_name": "MiniGrid-DoorKey-5x5-v0", "grid_size": 5},
    "doorkey-6x6": {"env_name": "MiniGrid-DoorKey-6x6-v0", "grid_size": 6},
    "doorkey-8x8": {"env_name": "MiniGrid-DoorKey-8x8-v0", "grid_size": 8},
}


class DynamicFullyObsWrapper(gym.ObservationWrapper):
    """Full grid observation instead of partial view."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.grid_size = grid_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (grid_size, grid_size, 3), np.uint8),
            'direction': spaces.Discrete(4),
            'mission': spaces.Text(max_length=128)
        })

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([10, 0, env.agent_dir])
        return {'image': full_grid, 'direction': obs['direction'], 'mission': obs['mission']}


class DynamicFlatObsWrapper(gym.ObservationWrapper):
    """Flatten grid to 1D vector."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.grid_size = grid_size
        self.obs_size = grid_size * grid_size * 3 + 4
        self.observation_space = spaces.Box(0, 255, (self.obs_size,), np.float32)

    def observation(self, obs):
        if isinstance(obs, dict):
            image = obs['image'].flatten().astype(np.float32) / 255.0
            direction = obs['direction']
        else:
            image = obs.flatten().astype(np.float32) / 255.0
            direction = 0

        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[direction] = 1.0
        return np.concatenate([image, dir_onehot])


class DynamicInfoWrapper(gym.Wrapper):
    """Add agent position and grid size to info."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.grid_size = grid_size

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['agent_pos'] = self.unwrapped.agent_pos
        info['agent_dir'] = self.unwrapped.agent_dir
        info['grid_size'] = self.grid_size
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['agent_pos'] = self.unwrapped.agent_pos
        info['agent_dir'] = self.unwrapped.agent_dir
        info['grid_size'] = self.grid_size
        return obs, info


def make_env(env_key: str, seed: int = None) -> Tuple[gym.Env, Dict[str, Any]]:
    """Create wrapped MiniGrid environment."""
    if env_key not in ENVIRONMENTS:
        raise ValueError(f"Unknown env: {env_key}. Available: {list(ENVIRONMENTS.keys())}")

    config = ENVIRONMENTS[env_key]
    grid_size = config["grid_size"]

    env = gym.make(config["env_name"])
    env = DynamicInfoWrapper(env, grid_size)
    env = DynamicFullyObsWrapper(env, grid_size)
    env = DynamicFlatObsWrapper(env, grid_size)

    if seed is not None:
        env.reset(seed=seed)

    env_info = {
        'env_key': env_key,
        'grid_size': grid_size,
        'obs_dim': grid_size * grid_size * 3 + 4,
        'num_actions': env.action_space.n,
        'disc_obs_dim': 6,
    }
    return env, env_info


def get_disc_obs(info: dict) -> np.ndarray:
    """Extract 6-dim discriminator obs: normalized (x, y) + direction one-hot."""
    pos = info['agent_pos']
    grid_size = info.get('grid_size', 8)

    x_norm = pos[0] / (grid_size - 1)
    y_norm = pos[1] / (grid_size - 1)

    dir_onehot = np.zeros(4, dtype=np.float32)
    dir_onehot[info['agent_dir']] = 1.0

    return np.array([x_norm, y_norm, *dir_onehot], dtype=np.float32)


if __name__ == "__main__":
    print("Testing environments...")
    for key in ENVIRONMENTS:
        try:
            env, info = make_env(key, seed=42)
            print(f"  {key}: {info['grid_size']}x{info['grid_size']}, obs={info['obs_dim']}")
            env.close()
        except Exception as e:
            print(f"  {key}: ERROR - {e}")
