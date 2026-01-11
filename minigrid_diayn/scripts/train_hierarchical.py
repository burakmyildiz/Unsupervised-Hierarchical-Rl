"""Hierarchical training: meta-controller over pre-trained DIAYN skills."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from core import DIAYNConfig, HierarchicalConfig, make_env, resolve_run, set_seed
from agents import DIAYNAgent, HierarchicalAgent


class GoalWrapper(gym.Wrapper):
    """Adds random goal and shaped reward."""

    def __init__(self, env, grid_size: int):
        super().__init__(env)
        self.grid_size = grid_size
        self.goal = None
        self.prev_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sample_goal(info)
        info["goal"] = self.goal
        info["goal_norm"] = self.goal / (self.grid_size - 1)
        self.prev_pos = np.array(info["agent_pos"], dtype=np.int64)
        self.prev_dist = self._get_distance(self.prev_pos)
        return obs, info

    def _sample_goal(self, info):
        agent_pos = tuple(info["agent_pos"])
        grid = self.env.unwrapped.grid

        walkable = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if (cell is None or cell.can_overlap()) and (x, y) != agent_pos:
                    walkable.append((x, y))

        reachable = self._reachable_positions(agent_pos)
        candidates = reachable if reachable else walkable
        self.goal = np.array(candidates[np.random.randint(len(candidates))])

    def _reachable_positions(self, start_pos):
        dist_map = self._distance_map_from(start_pos)
        reachable = []
        for x in range(dist_map.shape[0]):
            for y in range(dist_map.shape[1]):
                if np.isfinite(dist_map[x, y]) and (x, y) != start_pos:
                    reachable.append((x, y))
        return reachable

    def _is_walkable(self, cell) -> bool:
        if cell is None:
            return True
        return cell.can_overlap()

    def _distance_map_from(self, start_pos):
        from collections import deque

        grid = self.env.unwrapped.grid
        width, height = grid.width, grid.height
        dist = np.full((width, height), np.inf, dtype=np.float32)
        sx, sy = start_pos
        dist[sx, sy] = 0.0
        goal_pos = tuple(self.goal) if self.goal is not None else None

        q = deque()
        q.append((sx, sy))
        while q:
            x, y = q.popleft()
            next_dist = dist[x, y] + 1.0
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if not np.isfinite(dist[nx, ny]):
                    cell = grid.get(nx, ny)
                    if self._is_walkable(cell) or (goal_pos is not None and (nx, ny) == goal_pos):
                        dist[nx, ny] = next_dist
                        q.append((nx, ny))
        return dist

    def _get_distance(self, pos):
        dist_map = self._distance_map_from(tuple(self.goal))
        px, py = pos
        dist = dist_map[px, py]
        if np.isfinite(dist):
            return dist
        # Fallback to Euclidean if unreachable.
        return np.linalg.norm(pos - self.goal)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        info["goal"] = self.goal
        info["goal_norm"] = self.goal / (self.grid_size - 1)

        agent_pos = info["agent_pos"]
        reached = np.array_equal(agent_pos, self.goal)

        # Distance shaping
        dist = self._get_distance(agent_pos)
        prev_dist = self._get_distance(self.prev_pos) if self.prev_pos is not None else dist
        shaping = (prev_dist - dist) * 0.5
        self.prev_dist = dist
        self.prev_pos = np.array(agent_pos, dtype=np.int64)

        reward = 10.0 if reached else (-0.1 + shaping)
        done = reached or terminated or truncated
        info["goal_reached"] = reached

        return obs, reward, done, truncated, info


def get_meta_state(state, info, grid_size, encoder, device):
    """Meta state: encoder features + normalized (x, y) + direction one-hot."""
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        features = encoder(state_t).cpu().numpy().squeeze(0)

    pos_norm = info.get("position_normalized")
    if pos_norm is None:
        pos = np.array(info["agent_pos"], dtype=np.float32)
        pos_norm = pos / (grid_size - 1)

    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[int(info["agent_dir"])] = 1.0

    meta_state = np.concatenate([features, pos_norm, dir_oh]).astype(np.float32)
    return meta_state


def train_hierarchical(
    run_dir: Path,
    num_episodes: int = 500,
    seed: int = 42,
    skill_duration: int = 16,
    deterministic_low: bool = True,
):
    set_seed(seed)

    # Load DIAYN config and model
    config = DIAYNConfig.load(run_dir / "config.json")
    diayn_path = run_dir / "diayn" / "final_model.pt"

    env, env_info = make_env(config.env_key, seed=seed)
    env = GoalWrapper(env, env_info["grid_size"])
    grid_size = env_info["grid_size"]

    # Load DIAYN agent
    diayn_agent = DIAYNAgent.from_checkpoint(diayn_path, config)

    # Create hierarchical agent
    meta_obs_dim = config.feature_dim + 6
    hier_config = HierarchicalConfig(
        num_skills=config.num_skills,
        skill_duration=skill_duration,
        meta_obs_dim=meta_obs_dim,
    )
    hier_agent = HierarchicalAgent(
        hier_config,
        env_info["obs_dim"],
        env_info["num_actions"],
        diayn_agent.policy,
        pretrained_encoder=diayn_agent.encoder,
    )

    print("=" * 60)
    print(f"Hierarchical Training on {config.env_key}")
    print(f"DIAYN checkpoint: {diayn_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)

    # Setup output
    hier_dir = run_dir / "hierarchical"
    hier_dir.mkdir(exist_ok=True)

    metrics = {"rewards": [], "success_rate": [], "lengths": []}
    successes = []

    pbar = tqdm(range(num_episodes), desc="Hierarchical")
    for ep in pbar:
        state, info = env.reset()
        goal = info["goal_norm"]
        meta_state = get_meta_state(state, info, grid_size, diayn_agent.encoder, hier_agent.device)

        hier_agent.reset_skill_counter()
        skill = hier_agent.select_skill(meta_state, goal)

        segment_start = meta_state.copy()
        segment_reward = 0.0
        segment_discount = 1.0
        ep_reward = 0.0

        for step in range(100):
            action = hier_agent.select_action(state, skill, deterministic=deterministic_low)
            next_state, reward, done, _, info = env.step(action)
            next_meta = get_meta_state(
                next_state, info, grid_size, diayn_agent.encoder, hier_agent.device
            )

            ep_reward += reward
            segment_reward += segment_discount * reward
            segment_discount *= hier_config.gamma

            hier_agent.step_skill_counter()

            if hier_agent.should_reselect_skill() or done:
                hier_agent.store_transition(
                    segment_start, goal, skill, segment_reward, next_meta, float(done)
                )
                hier_agent.update()

                if not done and hier_agent.should_reselect_skill():
                    hier_agent.reset_skill_counter()
                    skill = hier_agent.select_skill(next_meta, goal)
                    segment_start = next_meta.copy()
                    segment_reward = 0.0
                    segment_discount = 1.0

            state = next_state
            if done:
                break

        successes.append(info.get("goal_reached", False))
        metrics["rewards"].append(ep_reward)
        metrics["lengths"].append(step + 1)

        if (ep + 1) % 50 == 0:
            rate = np.mean(successes[-50:])
            metrics["success_rate"].append(rate)
            pbar.set_postfix(reward=f"{ep_reward:.1f}", success=f"{rate:.1%}")

    # Save
    hier_agent.save(hier_dir / "final_model.pt")
    with open(hier_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    env.close()

    print("\n" + "=" * 60)
    print("Hierarchical Training Complete")
    print(f"Final success rate: {np.mean(successes[-100:]):.1%}")
    print(f"Saved to: {hier_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="latest", help="Run dir or 'latest'")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skill_duration", type=int, default=16)
    parser.add_argument("--stochastic_low", action="store_true",
                        help="Use stochastic low-level actions")
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    train_hierarchical(
        run_dir,
        num_episodes=args.episodes,
        seed=args.seed,
        skill_duration=args.skill_duration,
        deterministic_low=not args.stochastic_low,
    )
