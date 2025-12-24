"""Goal-reaching task for evaluating hierarchical skills."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class GoalReachingWrapper(gym.Wrapper):
    """Wrap environment with random goal and shaped reward."""

    def __init__(self, env, grid_size: int, goal_reward: float = 10.0, step_penalty: float = -0.1,
                 shaping_coef: float = 0.5):
        super().__init__(env)
        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.shaping_coef = shaping_coef
        self.goal_pos = None
        self.prev_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_goal(info)
        info['goal_pos'] = self.goal_pos
        info['goal_normalized'] = self._normalize_pos(self.goal_pos)
        # Initialize distance for shaping
        self.prev_dist = np.linalg.norm(info['agent_pos'] - self.goal_pos)
        return obs, info

    def _sample_goal(self, info):
        """Sample random goal position from walkable cells only."""
        agent_pos = tuple(info['agent_pos'])
        grid = self.env.unwrapped.grid

        # Collect all walkable positions (empty or can_overlap)
        walkable = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if (cell is None or cell.can_overlap()) and (x, y) != agent_pos:
                    walkable.append((x, y))

        # Sample from valid positions
        goal_pos = walkable[np.random.randint(len(walkable))]
        self.goal_pos = np.array(goal_pos)

    def _normalize_pos(self, pos) -> np.ndarray:
        """Normalize position to [0, 1]."""
        return pos / (self.grid_size - 1)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        agent_pos = info['agent_pos']
        info['goal_pos'] = self.goal_pos
        info['goal_normalized'] = self._normalize_pos(self.goal_pos)

        # Check goal reached
        reached = np.array_equal(agent_pos, self.goal_pos)

        # Distance-based reward shaping
        curr_dist = np.linalg.norm(agent_pos - self.goal_pos)
        shaping = (self.prev_dist - curr_dist) * self.shaping_coef
        self.prev_dist = curr_dist

        reward = self.goal_reward if reached else (self.step_penalty + shaping)
        done = reached or terminated or truncated

        info['goal_reached'] = reached
        return obs, reward, done, truncated, info

    def get_goal(self) -> np.ndarray:
        """Get normalized goal position for meta-controller input."""
        return self._normalize_pos(self.goal_pos)


def make_goal_env(env_key: str, seed: int = None) -> Tuple[gym.Env, dict]:
    """Create goal-reaching environment."""
    from environments import make_env, ENVIRONMENTS

    env, env_info = make_env(env_key, seed)
    grid_size = env_info['grid_size']

    env = GoalReachingWrapper(env, grid_size)
    env_info['goal_dim'] = 2

    return env, env_info


def train_hierarchical(
    diayn_checkpoint: str,
    env_key: str = "empty-8x8",
    num_episodes: int = 500,
    max_steps: int = 100,
    seed: int = 42,
    save_dir: str = "checkpoints_hierarchical"
):
    """Train hierarchical agent on goal-reaching task."""
    import os
    import torch
    from diayn_agent import DIAYNAgent
    from hierarchical_agent import HierarchicalAgent, HierarchicalConfig
    from config import get_config

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load pre-trained DIAYN
    env, env_info = make_goal_env(env_key, seed)
    grid_size = env_info['grid_size']
    diayn_config = get_config()
    diayn_agent = DIAYNAgent(diayn_config, env_info['obs_dim'], env_info['num_actions'], grid_size=grid_size)
    diayn_agent.load(diayn_checkpoint)

    # Create hierarchical agent
    hier_config = HierarchicalConfig(
        num_skills=diayn_config.num_skills,
        goal_dim=env_info['goal_dim'],
    )
    hier_agent = HierarchicalAgent(
        hier_config,
        env_info['obs_dim'],
        env_info['num_actions'],
        diayn_agent.policy
    )

    # Training loop
    metrics = {'episode_rewards': [], 'success_rate': [], 'episode_lengths': []}
    successes = []
    grid_size = env_info['grid_size']

    def get_meta_state(info):
        """Get simplified state for meta-controller: position + direction one-hot (6D)."""
        pos_norm = np.array(info['agent_pos'], dtype=np.float32) / (grid_size - 1)
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[info['agent_dir']] = 1.0
        return np.concatenate([pos_norm, dir_onehot])

    for ep in range(num_episodes):
        state, info = env.reset()
        goal = env.get_goal()
        meta_state = get_meta_state(info)
        episode_reward = 0
        hier_agent.reset_skill_counter()
        skill = hier_agent.select_skill(meta_state, goal)

        # Track segment state and reward with proper discounting
        segment_start_meta = meta_state.copy()
        segment_reward = 0.0
        segment_discount = 1.0
        gamma = hier_config.gamma

        for step in range(max_steps):
            action = hier_agent.select_action(state, skill)  # low-level needs full state
            next_state, reward, done, _, info = env.step(action)
            next_meta_state = get_meta_state(info)
            episode_reward += reward

            # Accumulate discounted reward within segment
            segment_reward += segment_discount * reward
            segment_discount *= gamma

            hier_agent.step_skill_counter()

            # Store transition at skill boundaries or episode end
            if hier_agent.should_reselect_skill() or done:
                hier_agent.store_transition(segment_start_meta, goal, skill, segment_reward, next_meta_state, float(done))
                hier_agent.update()

                if not done and hier_agent.should_reselect_skill():
                    hier_agent.reset_skill_counter()
                    skill = hier_agent.select_skill(next_meta_state, goal)
                    segment_start_meta = next_meta_state.copy()
                    segment_reward = 0.0
                    segment_discount = 1.0

            state = next_state
            meta_state = next_meta_state
            if done:
                break

        successes.append(info.get('goal_reached', False))
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)

        if (ep + 1) % 50 == 0:
            recent_success = np.mean(successes[-50:])
            metrics['success_rate'].append(recent_success)
            print(f"Episode {ep+1}: reward={episode_reward:.1f}, success_rate={recent_success:.2%}")

    # Save
    hier_agent.save(os.path.join(save_dir, "hierarchical_final.pt"))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pt")
    parser.add_argument("--env", default="empty-8x8")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    train_hierarchical(args.checkpoint, args.env, args.episodes)
