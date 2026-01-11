"""MiniGrid environment creation and wrappers for DIAYN skill discovery."""

from typing import Dict, Tuple

import gymnasium as gym
import minigrid  # registers envs
import numpy as np
from gymnasium import spaces


ENVIRONMENTS = {
    "empty-6x6": {"name": "MiniGrid-Empty-6x6-v0", "grid_size": 6},
    "empty-8x8": {"name": "MiniGrid-Empty-8x8-v0", "grid_size": 8},
    "empty-16x16": {"name": "MiniGrid-Empty-16x16-v0", "grid_size": 16},
    "fourrooms": {"name": "MiniGrid-FourRooms-v0", "grid_size": 19},
    "doorkey-5x5": {"name": "MiniGrid-DoorKey-5x5-v0", "grid_size": 5},
    "doorkey-6x6": {"name": "MiniGrid-DoorKey-6x6-v0", "grid_size": 6},
    "doorkey-8x8": {"name": "MiniGrid-DoorKey-8x8-v0", "grid_size": 8},
}

# Reverse mapping: full name -> key
_NAME_TO_KEY = {v["name"]: k for k, v in ENVIRONMENTS.items()}


def resolve_env_key(env: str) -> str:
    """Convert env name or key to canonical key."""
    if env in ENVIRONMENTS:
        return env
    if env in _NAME_TO_KEY:
        return _NAME_TO_KEY[env]
    raise ValueError(f"Unknown env: {env}. Available: {list(ENVIRONMENTS.keys())}")


class MovementOnlyWrapper(gym.ActionWrapper):
    """Restrict to movement actions only for skill discovery.

    MiniGrid has 7 actions but only 3 affect movement:
    - 0: turn left
    - 1: turn right
    - 2: move forward

    Actions 3-6 (pickup, drop, toggle, done) are no-ops for Empty/FourRooms
    and cause the "camping" equilibrium where agents learn not to move.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=right, 2=forward

    def action(self, action):
        # Actions 0, 1, 2 map directly to MiniGrid's left, right, forward
        return action


class RandomStartWrapper(gym.Wrapper):
    """Randomize agent starting position for skill diversity.

    Without this, all episodes start from the same position, making it hard
    for skills to differentiate in early trajectory steps.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        # First do normal reset to set up the grid
        obs, info = self.env.reset(**kwargs)

        # Find valid positions (empty cells, not walls)
        env = self.unwrapped
        valid_positions = []
        for x in range(1, env.grid.width - 1):
            for y in range(1, env.grid.height - 1):
                cell = env.grid.get(x, y)
                # Empty cell or floor tile
                if cell is None:
                    valid_positions.append((x, y))
                elif hasattr(cell, 'type') and cell.type == 'floor':
                    valid_positions.append((x, y))

        # Randomize position if we found valid spots
        if valid_positions:
            new_pos = valid_positions[np.random.randint(len(valid_positions))]
            env.agent_pos = np.array(new_pos)
            env.agent_dir = np.random.randint(4)  # Random direction too

            # CRITICAL: Regenerate observation with new position
            # This updates obs["direction"] and obs["image"] for partial view
            obs = env.gen_obs()

        return obs, info


class InfoWrapper(gym.Wrapper):
    """Add agent position, direction, and grid size to info."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.grid_size = grid_size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.update(self._agent_info())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self._agent_info())
        return obs, reward, terminated, truncated, info

    def _agent_info(self) -> dict:
        pos = self.unwrapped.agent_pos
        return {
            "agent_pos": pos,
            "agent_dir": self.unwrapped.agent_dir,
            "grid_size": self.grid_size,
            "position_normalized": np.array([
                pos[0] / (self.grid_size - 1),
                pos[1] / (self.grid_size - 1)
            ], dtype=np.float32),
        }


class FullyObsWrapper(gym.ObservationWrapper):
    """Full grid observation instead of partial 7x7 view."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.grid_size = grid_size
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (grid_size, grid_size, 3), np.uint8),
            "direction": spaces.Discrete(4),
            "mission": spaces.Text(max_length=128),
        })

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([10, 0, env.agent_dir])
        return {"image": full_grid, "direction": obs["direction"], "mission": obs["mission"]}


class FlatObsWrapper(gym.ObservationWrapper):
    """Flatten grid to 1D vector with direction one-hot."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.obs_dim = grid_size * grid_size * 3 + 4
        self.observation_space = spaces.Box(0, 1, (self.obs_dim,), np.float32)

    def observation(self, obs):
        if isinstance(obs, dict):
            # NO division by 255 - MiniGrid uses symbolic encoding (0-10)
            image = obs["image"].flatten().astype(np.float32)
            direction = obs["direction"]
        else:
            image = obs.flatten().astype(np.float32)
            direction = 0

        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[direction] = 1.0
        return np.concatenate([image, dir_onehot])


class PartialFlatObsWrapper(gym.ObservationWrapper):
    """Flatten partial 7x7 observation (MiniGrid default) to 1D vector."""

    def __init__(self, env):
        super().__init__(env)
        # MiniGrid default: 7x7x3 image
        self.obs_dim = 7 * 7 * 3
        self.observation_space = spaces.Box(0, 1, (self.obs_dim,), np.float32)

    def observation(self, obs):
        if isinstance(obs, dict):
            image = obs["image"]
        else:
            image = obs
        # Flatten to float32 - NO division by 255!
        # MiniGrid uses symbolic encoding (0-10), not RGB (0-255)
        return image.flatten().astype(np.float32)


def make_env(env_key: str, seed: int = None, partial_obs: bool = False,
             random_start: bool = True, movement_only: bool = False) -> Tuple[gym.Env, Dict]:
    """Create wrapped MiniGrid environment.

    Args:
        env_key: Short key (e.g., 'fourrooms') or full name
        seed: Random seed
        partial_obs: If True, use partial 7x7 agent-centric view.
                     If False, use full grid observation.
        random_start: If True, randomize agent starting position each episode.
                      Critical for skill diversity - without this, all episodes
                      start from same position.
        movement_only: If True, restrict actions to {left, right, forward}.
                       Removes no-op actions that cause "camping" equilibrium.

    Returns:
        (env, env_info) where env_info contains obs_dim, num_actions, grid_size, etc.
    """
    key = resolve_env_key(env_key)
    cfg = ENVIRONMENTS[key]
    grid_size = cfg["grid_size"]

    env = gym.make(cfg["name"])

    # MovementOnlyWrapper should come first (before RandomStart)
    # so action space is restricted before any position changes
    if movement_only:
        env = MovementOnlyWrapper(env)

    # RandomStartWrapper MUST come before observation wrappers
    # so they see the updated agent position
    if random_start:
        env = RandomStartWrapper(env)

    env = InfoWrapper(env, grid_size)

    if partial_obs:
        # Use MiniGrid's default 7x7 partial view (agent-centric)
        env = PartialFlatObsWrapper(env)
        obs_dim = 7 * 7 * 3  # 147
        view_size = 7
    else:
        # Use full grid observation (global view)
        env = FullyObsWrapper(env, grid_size)
        env = FlatObsWrapper(env, grid_size)
        obs_dim = grid_size * grid_size * 3 + 4
        view_size = grid_size

    if seed is not None:
        env.reset(seed=seed)

    env_info = {
        "env_key": key,
        "env_name": cfg["name"],
        "grid_size": grid_size,
        "view_size": view_size,
        "obs_dim": obs_dim,
        "num_actions": env.action_space.n,
        "partial_obs": partial_obs,
        "random_start": random_start,
        "movement_only": movement_only,
    }
    return env, env_info
