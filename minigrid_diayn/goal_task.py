"""Goal-reaching task for evaluating hierarchical skills."""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class GoalReachingWrapper(gym.Wrapper):
    """Wrap environment with random goal and sparse reward."""

    def __init__(self, env, grid_size: int, goal_reward: float = 10.0, step_penalty: float = -0.1):
        super().__init__(env)
        self.grid_size = grid_size
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.goal_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_goal(info)
        info['goal_pos'] = self.goal_pos
        info['goal_normalized'] = self._normalize_pos(self.goal_pos)
        return obs, info

    def _sample_goal(self, info):
        """Sample random goal position different from agent start."""
        agent_pos = info['agent_pos']
        while True:
            # Sample valid position (avoiding walls at edges)
            x = np.random.randint(1, self.grid_size - 1)
            y = np.random.randint(1, self.grid_size - 1)
            if (x, y) != tuple(agent_pos):
                self.goal_pos = np.array([x, y])
                break

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
        reward = self.goal_reward if reached else self.step_penalty
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
    diayn_config = get_config()
    diayn_agent = DIAYNAgent(diayn_config, env_info['obs_dim'], env_info['num_actions'])
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

    for ep in range(num_episodes):
        state, info = env.reset()
        goal = env.get_goal()
        episode_reward = 0
        hier_agent.reset_skill_counter()
        skill = hier_agent.select_skill(state, goal)

        for step in range(max_steps):
            action = hier_agent.select_action(state, skill)
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward

            hier_agent.step_skill_counter()

            # Store transition at skill boundaries or episode end
            if hier_agent.should_reselect_skill() or done:
                hier_agent.store_transition(state, goal, skill, episode_reward, next_state, float(done))
                hier_agent.update()

                if not done and hier_agent.should_reselect_skill():
                    hier_agent.reset_skill_counter()
                    skill = hier_agent.select_skill(next_state, goal)

            state = next_state
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
