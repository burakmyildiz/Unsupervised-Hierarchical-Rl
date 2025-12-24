"""Multi-seed experiment runner for DIAYN with statistical aggregation."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import torch
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    env_key: str = "empty-8x8"
    num_skills: int = 8
    num_episodes: int = 2000
    max_steps: int = 100
    seeds: List[int] = None
    alpha: float = 0.1

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1000]


def run_single_seed(config: ExperimentConfig, seed: int, output_dir: str) -> Dict:
    """Run training for a single seed."""
    from environments import make_env
    from diayn_agent import DIAYNAgent
    from config import get_config

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup
    env, env_info = make_env(config.env_key, seed=seed)
    diayn_config = get_config(
        num_skills=config.num_skills,
        num_episodes=config.num_episodes,
        max_steps=config.max_steps,
        alpha=config.alpha,
        seed=seed
    )

    agent = DIAYNAgent(diayn_config, env_info['obs_dim'], env_info['num_actions'])

    # Training metrics
    metrics = {
        'episode_rewards': [],
        'discriminator_accuracy': [],
        'entropy': [],
    }

    total_steps = 0
    for ep in range(config.num_episodes):
        skill = np.random.randint(config.num_skills)
        state, info = env.reset()
        episode_reward = 0

        for step in range(config.max_steps):
            if total_steps < 1000:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, skill)

            next_state, _, terminated, truncated, next_info = env.step(action)
            reward = agent.compute_pseudo_reward(next_info, skill)
            episode_reward += reward

            disc_obs = agent.get_discriminator_obs(next_info)
            agent.replay_buffer.push(state, action, reward, next_state,
                                    float(terminated or truncated), skill, disc_obs)

            if total_steps >= 1000:
                train_metrics = agent.train_step(diayn_config.batch_size)
                if train_metrics:
                    metrics['discriminator_accuracy'].append(train_metrics['discriminator_accuracy'])
                    metrics['entropy'].append(train_metrics['entropy'])

            state = next_state
            total_steps += 1

            if terminated or truncated:
                break

        metrics['episode_rewards'].append(episode_reward)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-100:])
            print(f"  Seed {seed} | Ep {ep+1}: avg_reward={avg_reward:.1f}")

    # Save checkpoint
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    agent.save(os.path.join(seed_dir, "final_model.pt"))

    with open(os.path.join(seed_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f)

    env.close()
    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across seeds with mean and std."""
    keys = all_metrics[0].keys()
    aggregated = {}

    for key in keys:
        arrays = [np.array(m[key]) for m in all_metrics]
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]
        stacked = np.stack(arrays)

        aggregated[key] = {
            'mean': stacked.mean(axis=0).tolist(),
            'std': stacked.std(axis=0).tolist(),
            'min': stacked.min(axis=0).tolist(),
            'max': stacked.max(axis=0).tolist(),
        }

    return aggregated


def plot_with_error_bars(aggregated: Dict, output_dir: str):
    """Create publication-ready plots with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Episode rewards
    ax = axes[0]
    mean = np.array(aggregated['episode_rewards']['mean'])
    std = np.array(aggregated['episode_rewards']['std'])
    x = np.arange(len(mean))

    mean_smooth = smooth(mean)
    std_smooth = smooth(std)
    x_smooth = np.arange(len(mean_smooth))

    ax.plot(x_smooth, mean_smooth, color='blue', linewidth=2)
    ax.fill_between(x_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth,
                   alpha=0.3, color='blue')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Pseudo-Reward')
    ax.set_title('Episode Rewards (mean ± std)')

    # Discriminator accuracy
    ax = axes[1]
    if 'discriminator_accuracy' in aggregated:
        mean = np.array(aggregated['discriminator_accuracy']['mean'])
        std = np.array(aggregated['discriminator_accuracy']['std'])
        mean_smooth = smooth(mean)
        std_smooth = smooth(std)
        x_smooth = np.arange(len(mean_smooth))

        ax.plot(x_smooth, mean_smooth, color='green', linewidth=2)
        ax.fill_between(x_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth,
                       alpha=0.3, color='green')
        ax.axhline(y=1/8, color='red', linestyle='--', label='Random')
        ax.set_ylim([0, 1])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Discriminator Accuracy')
    ax.legend()

    # Entropy
    ax = axes[2]
    if 'entropy' in aggregated:
        mean = np.array(aggregated['entropy']['mean'])
        std = np.array(aggregated['entropy']['std'])
        mean_smooth = smooth(mean)
        std_smooth = smooth(std)
        x_smooth = np.arange(len(mean_smooth))

        ax.plot(x_smooth, mean_smooth, color='purple', linewidth=2)
        ax.fill_between(x_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth,
                       alpha=0.3, color='purple')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Aggregated plots saved to {output_dir}/aggregated_curves.png")


def run_experiment(config: ExperimentConfig, output_dir: str = None):
    """Run full experiment with multiple seeds."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"experiments/{config.name}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Running experiment: {config.name}")
    print(f"Seeds: {config.seeds}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    all_metrics = []
    for seed in config.seeds:
        print(f"\n--- Seed {seed} ---")
        metrics = run_single_seed(config, seed, output_dir)
        all_metrics.append(metrics)

    # Aggregate and plot
    aggregated = aggregate_metrics(all_metrics)
    with open(os.path.join(output_dir, "aggregated_metrics.json"), 'w') as f:
        json.dump(aggregated, f)

    plot_with_error_bars(aggregated, output_dir)

    print(f"\n{'='*60}")
    print(f"Experiment complete: {config.name}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return aggregated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--env", default="empty-8x8")
    parser.add_argument("--skills", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    config = ExperimentConfig(
        name=args.name,
        env_key=args.env,
        num_skills=args.skills,
        num_episodes=args.episodes,
        seeds=args.seeds,
    )
    run_experiment(config)
