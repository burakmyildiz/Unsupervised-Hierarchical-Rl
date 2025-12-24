"""
Training script for DIAYN on MiniGrid environments.

Trains an agent to discover diverse skills without external rewards.
"""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
from datetime import datetime

from config import DIAYNConfig, get_config
from wrappers import make_env, get_env_info
from diayn_agent import DIAYNAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_diayn(config: DIAYNConfig) -> Tuple[DIAYNAgent, Dict[str, List]]:
    """
    Main training loop for DIAYN.

    Args:
        config: Training configuration

    Returns:
        agent: Trained DIAYN agent
        metrics: Dictionary of training metrics
    """
    # 1. Seeds are set
    set_seed(config.seed)

    # 2. Environment is created
    env = make_env(config.env_name, fully_observable=True, seed=config.seed)
    env_info = get_env_info(env)

    print("=" * 60)
    print("DIAYN Training Configuration")
    print("=" * 60)
    print(f"Environment: {config.env_name}")
    print(f"Observation dim: {env_info['obs_dim']}")
    print(f"Number of actions: {env_info['num_actions']}")
    print(f"Number of skills: {config.num_skills}")
    print(f"Device: {config.device}")
    print(f"Total episodes: {config.num_episodes}")
    print(f"Max steps per episode: {config.max_steps}")
    print("=" * 60)

    # 3. Agent is created (pass grid_size for proper position normalization)
    grid_size = env.unwrapped.width
    print(f"Grid size: {grid_size}")

    agent = DIAYNAgent(
        config=config,
        obs_dim=env_info['obs_dim'],
        num_actions=env_info['num_actions'],
        grid_size=grid_size
    )

    # 4. Metrics storage
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'discriminator_loss': [],
        'discriminator_accuracy': [],
        'q1_loss': [],
        'q2_loss': [],
        'policy_loss': [],
        'entropy': [],
        'alpha': [],
        'skill_distribution': [],
    }

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)

    # 5. Training loop
    total_steps = 0
    best_disc_accuracy = 0.0

    pbar = tqdm(range(config.num_episodes), desc="Training")

    for episode in pbar:
        # 5.1. Randomly sample a skill
        skill = np.random.randint(0, config.num_skills)

        # 5.2. Reset environment
        state, info = env.reset()

        episode_reward = 0.0
        episode_length = 0

        # 5.3. Episode loop
        for step in range(config.max_steps):
            # 5.3.1. Select action before training starts
            if total_steps < config.start_training_step:
                # 5.3.1.1. Random action for initial exploration
                action = env.action_space.sample()
            else:
                # 5.3.1.2. Select from policy
                action = agent.select_action(state, skill, deterministic=False)

            # 5.3.2. Step environment and observe
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Note that env_reward is ignored in DIAYN

            # 5.3.3. Compute pseudo-reward
            # Formula: log q(skill | state) - log p(skill)
            # probability that skill produced the state minus prior probability of skill
            # Get discriminator observation (6-dim: x, y, direction)
            disc_next_obs = agent.get_discriminator_obs(info)
            pseudo_reward = agent.compute_pseudo_reward(info, skill)

            # 5.3.4. Store transition in replay buffer
            # store skill along with transition to train discriminator
            agent.replay_buffer.push(
                state, action, pseudo_reward, next_state, done, skill, disc_next_obs
            )

            # 5.3.5. Update statistics
            episode_reward += pseudo_reward
            episode_length += 1
            total_steps += 1

            # 5.3.6. Train agent
            if total_steps >= config.start_training_step:
                for _ in range(config.updates_per_step):
                    train_metrics = agent.train_step(config.batch_size)

            # 5.3.7. Transition to next state
            state = next_state

            if done:
                break

        # 5.4. Store episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['skill_distribution'].append(skill)

        # 5.5. Store training metrics
        if total_steps >= config.start_training_step:
            metrics['discriminator_loss'].append(train_metrics['discriminator_loss'])
            metrics['discriminator_accuracy'].append(train_metrics['discriminator_accuracy'])
            metrics['q1_loss'].append(train_metrics['q1_loss'])
            metrics['q2_loss'].append(train_metrics['q2_loss'])
            metrics['policy_loss'].append(train_metrics['policy_loss'])
            metrics['entropy'].append(train_metrics['entropy'])
            metrics['alpha'].append(train_metrics['alpha'])

        # 5.6. Logging
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-config.log_interval:])

            if total_steps >= config.start_training_step:
                avg_disc_acc = np.mean(metrics['discriminator_accuracy'][-config.log_interval:])
                avg_entropy = np.mean(metrics['entropy'][-config.log_interval:])

                pbar.set_postfix({
                    'reward': f'{avg_reward:.2f}',
                    'disc_acc': f'{avg_disc_acc:.3f}',
                    'entropy': f'{avg_entropy:.3f}',
                    'alpha': f'{train_metrics["alpha"]:.3f}'
                })

                # Track best discriminator accuracy
                if avg_disc_acc > best_disc_accuracy:
                    best_disc_accuracy = avg_disc_acc
            else:
                pbar.set_postfix({
                    'reward': f'{avg_reward:.2f}',
                    'status': 'exploring'
                })

        # 5.7. Save checkpoint
        if (episode + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.save_dir, f'checkpoint_ep{episode+1}.pt'
            )
            agent.save(checkpoint_path)

    # 5.8. Save final model
    final_path = os.path.join(config.save_dir, 'final_model.pt')
    agent.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    # 5.9. Save metrics
    metrics_path = os.path.join(config.save_dir, 'metrics.json')
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")

    # 5.10. Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total episodes: {config.num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Best discriminator accuracy: {best_disc_accuracy:.4f}")
    print(f"Final average reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    print("=" * 60)

    env.close()
    return agent, metrics


def save_metrics(metrics: Dict[str, List], path: str):
    """Save metrics to JSON file."""
    # 1. Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value

    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def load_metrics(path: str) -> Dict[str, List]:
    """Load metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train DIAYN on MiniGrid')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                        help='Environment name')
    parser.add_argument('--num_skills', type=int, default=8,
                        help='Number of skills to discover')
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu, cuda, mps)')

    args = parser.parse_args()

    config_kwargs = {
        'env_name': args.env,
        'num_skills': args.num_skills,
        'num_episodes': args.num_episodes,
        'seed': args.seed,
    }

    if args.device is not None:
        config_kwargs['device'] = args.device

    config = get_config(**config_kwargs)

    # Train
    agent, metrics = train_diayn(config)

    print("\nTo visualize results, run:")
    print(f"  python visualize.py --checkpoint {config.save_dir}/final_model.pt")
