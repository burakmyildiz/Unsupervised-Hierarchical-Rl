"""
Evaluation functions for DIAYN skills.

Computes metrics for skill diversity, consistency, and state coverage.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

from config import DIAYNConfig
from wrappers import make_env, get_env_info
from diayn_agent import DIAYNAgent


def evaluate_skill_diversity(
    agent: DIAYNAgent,
    env,
    num_episodes_per_skill: int = 10,
    max_steps: int = 100
) -> Dict[str, float]:
    """
    Evaluate diversity of learned skills.

    Metrics:
    - State coverage: How much of the grid is visited across all skills
    - Final position spread: How different are final positions across skills
    - Trajectory length variance: Do skills have different episode lengths

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes_per_skill: Episodes to run per skill
        max_steps: Maximum steps per episode

    Returns:
        Dictionary of diversity metrics
    """
    num_skills = agent.num_skills

    # Collect trajectories for each skill
    all_positions = defaultdict(list)  # skill -> list of visited positions
    final_positions = defaultdict(list)  # skill -> list of final positions
    episode_lengths = defaultdict(list)  # skill -> list of episode lengths

    for skill in range(num_skills):
        for _ in range(num_episodes_per_skill):
            state, info = env.reset()
            positions = [tuple(info['agent_pos'])]

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)
                positions.append(tuple(info['agent_pos']))

                if terminated or truncated:
                    break

            all_positions[skill].extend(positions)
            final_positions[skill].append(positions[-1])
            episode_lengths[skill].append(len(positions))

    # Compute metrics
    metrics = {}

    # 1. State coverage per skill
    unique_states_per_skill = []
    for skill in range(num_skills):
        unique = len(set(all_positions[skill]))
        unique_states_per_skill.append(unique)
    metrics['avg_unique_states_per_skill'] = np.mean(unique_states_per_skill)
    metrics['std_unique_states_per_skill'] = np.std(unique_states_per_skill)

    # 2. Total state coverage (across all skills)
    all_visited = set()
    for positions in all_positions.values():
        all_visited.update(positions)
    grid_size = env.unwrapped.width * env.unwrapped.height
    metrics['total_state_coverage'] = len(all_visited) / grid_size

    # 3. Final position diversity
    all_final = []
    for skill in range(num_skills):
        all_final.extend(final_positions[skill])

    # Compute pairwise distances between skill final positions
    skill_centroids = []
    for skill in range(num_skills):
        if final_positions[skill]:
            centroid = np.mean(final_positions[skill], axis=0)
            skill_centroids.append(centroid)

    if len(skill_centroids) >= 2:
        # Average distance between skill centroids
        distances = []
        for i in range(len(skill_centroids)):
            for j in range(i + 1, len(skill_centroids)):
                d = np.linalg.norm(
                    np.array(skill_centroids[i]) - np.array(skill_centroids[j])
                )
                distances.append(d)
        metrics['avg_centroid_distance'] = np.mean(distances)
    else:
        metrics['avg_centroid_distance'] = 0.0

    # 4. Episode length variance across skills
    avg_lengths = [np.mean(episode_lengths[s]) for s in range(num_skills)]
    metrics['episode_length_variance'] = np.var(avg_lengths)

    return metrics


def evaluate_skill_consistency(
    agent: DIAYNAgent,
    env,
    num_episodes: int = 20,
    max_steps: int = 100
) -> Dict[str, float]:
    """
    Evaluate consistency of learned skills.

    A good skill should produce similar trajectories when run multiple times.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes: Episodes to run per skill
        max_steps: Maximum steps per episode

    Returns:
        Dictionary of consistency metrics
    """
    num_skills = agent.num_skills

    # Collect final positions for each skill
    final_positions = defaultdict(list)

    for skill in range(num_skills):
        for _ in range(num_episodes):
            state, info = env.reset()

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            final_positions[skill].append(info['agent_pos'])

    # Compute consistency metrics
    metrics = {}

    # Variance of final positions per skill (lower = more consistent)
    variances = []
    for skill in range(num_skills):
        if len(final_positions[skill]) > 1:
            positions = np.array(final_positions[skill])
            var = np.mean(np.var(positions, axis=0))
            variances.append(var)

    metrics['avg_final_position_variance'] = np.mean(variances) if variances else 0.0
    metrics['consistency_score'] = 1.0 / (1.0 + metrics['avg_final_position_variance'])

    return metrics


def evaluate_discriminator_accuracy(
    agent: DIAYNAgent,
    env,
    num_episodes_per_skill: int = 10,
    max_steps: int = 100
) -> Dict[str, float]:
    """
    Evaluate how well the discriminator can predict skills.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes_per_skill: Episodes to run per skill
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with discriminator accuracy metrics
    """
    num_skills = agent.num_skills

    correct = 0
    total = 0
    confusion_matrix = np.zeros((num_skills, num_skills), dtype=np.int32)

    for true_skill in range(num_skills):
        for _ in range(num_episodes_per_skill):
            state, info = env.reset()

            for step in range(max_steps):
                action = agent.select_action(state, true_skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)

                # Predict skill from state
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    predicted = agent.discriminator.predict_skill(state_tensor).item()

                confusion_matrix[true_skill, predicted] += 1

                if predicted == true_skill:
                    correct += 1
                total += 1

                if terminated or truncated:
                    break

    metrics = {
        'overall_accuracy': correct / total if total > 0 else 0.0,
        'confusion_matrix': confusion_matrix.tolist(),
    }

    # Per-skill accuracy
    per_skill_acc = []
    for skill in range(num_skills):
        skill_total = confusion_matrix[skill].sum()
        if skill_total > 0:
            acc = confusion_matrix[skill, skill] / skill_total
            per_skill_acc.append(acc)
    metrics['per_skill_accuracy'] = per_skill_acc
    metrics['accuracy_std'] = np.std(per_skill_acc) if per_skill_acc else 0.0

    return metrics


def full_evaluation(
    agent: DIAYNAgent,
    config: DIAYNConfig,
    num_episodes_per_skill: int = 10
) -> Dict[str, float]:
    """
    Run full evaluation suite.

    Args:
        agent: Trained DIAYN agent
        config: Configuration
        num_episodes_per_skill: Episodes to run per skill

    Returns:
        Combined dictionary of all metrics
    """
    env = make_env(config.env_name, fully_observable=True, seed=config.seed + 1000)

    print("\nEvaluating skill diversity...")
    diversity_metrics = evaluate_skill_diversity(
        agent, env, num_episodes_per_skill, config.max_steps
    )

    print("Evaluating skill consistency...")
    consistency_metrics = evaluate_skill_consistency(
        agent, env, num_episodes_per_skill * 2, config.max_steps
    )

    print("Evaluating discriminator accuracy...")
    disc_metrics = evaluate_discriminator_accuracy(
        agent, env, num_episodes_per_skill, config.max_steps
    )

    env.close()

    # Combine all metrics
    all_metrics = {
        **diversity_metrics,
        **consistency_metrics,
        **disc_metrics
    }

    return all_metrics


def print_evaluation_report(metrics: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("DIAYN EVALUATION REPORT")
    print("=" * 60)

    print("\n--- Skill Diversity ---")
    print(f"  Total state coverage: {metrics['total_state_coverage']:.1%}")
    print(f"  Avg unique states per skill: {metrics['avg_unique_states_per_skill']:.1f}")
    print(f"  Avg centroid distance: {metrics['avg_centroid_distance']:.2f}")

    print("\n--- Skill Consistency ---")
    print(f"  Consistency score: {metrics['consistency_score']:.3f}")
    print(f"  Final position variance: {metrics['avg_final_position_variance']:.3f}")

    print("\n--- Discriminator Performance ---")
    print(f"  Overall accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"  Accuracy std across skills: {metrics['accuracy_std']:.3f}")

    if 'per_skill_accuracy' in metrics:
        print("  Per-skill accuracy:")
        for i, acc in enumerate(metrics['per_skill_accuracy']):
            print(f"    Skill {i}: {acc:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    from config import get_config

    parser = argparse.ArgumentParser(description='Evaluate DIAYN agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                        help='Environment name')
    parser.add_argument('--num_skills', type=int, default=8,
                        help='Number of skills')
    parser.add_argument('--episodes_per_skill', type=int, default=10,
                        help='Episodes per skill for evaluation')

    args = parser.parse_args()

    # Load config and agent
    config = get_config(env_name=args.env, num_skills=args.num_skills)
    env = make_env(config.env_name, fully_observable=True)
    env_info = get_env_info(env)

    agent = DIAYNAgent(
        config=config,
        obs_dim=env_info['obs_dim'],
        num_actions=env_info['num_actions']
    )
    agent.load(args.checkpoint)
    env.close()

    # Run evaluation
    metrics = full_evaluation(agent, config, args.episodes_per_skill)
    print_evaluation_report(metrics)
