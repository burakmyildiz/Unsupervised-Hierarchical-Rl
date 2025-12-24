"""Visualize hierarchical controller using pre-trained skills."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

from environments import make_env
from diayn_agent import DIAYNAgent
from hierarchical_agent import HierarchicalAgent, HierarchicalConfig
from config import get_config
from visualize import draw_env_grid


def get_valid_goals(env, num_goals=5):
    """Get valid walkable goal positions from different areas of the grid."""
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height

    # Collect walkable positions
    walkable = []
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is None or cell.can_overlap():
                walkable.append((x, y))

    if len(walkable) < num_goals:
        return walkable

    # Sample from different quadrants for diversity
    quadrants = [[], [], [], [], []]  # TL, TR, BL, BR, center
    mid_x, mid_y = width // 2, height // 2

    for pos in walkable:
        x, y = pos
        if x < mid_x and y < mid_y:
            quadrants[0].append(pos)
        elif x >= mid_x and y < mid_y:
            quadrants[1].append(pos)
        elif x < mid_x and y >= mid_y:
            quadrants[2].append(pos)
        elif x >= mid_x and y >= mid_y:
            quadrants[3].append(pos)
        if abs(x - mid_x) <= 2 and abs(y - mid_y) <= 2:
            quadrants[4].append(pos)

    goals = []
    for q in quadrants:
        if q and len(goals) < num_goals:
            goals.append(q[np.random.randint(len(q))])

    return goals[:num_goals]


def visualize_skill_selection(
    diayn_checkpoint: str,
    env_key: str = "fourrooms",
    num_episodes: int = 10,
    max_steps: int = 100,
    output_dir: str = "plots_hierarchical"
):
    """Visualize which skills would be useful for different goals."""
    os.makedirs(output_dir, exist_ok=True)

    # Load environment and agent
    env, env_info = make_env(env_key, seed=42)
    config = get_config(num_skills=8)

    agent = DIAYNAgent(config, env_info['obs_dim'], env_info['num_actions'])
    agent.load(diayn_checkpoint)

    grid_size = env_info['grid_size']

    # Test each skill's ability to reach different areas
    print("Testing skill coverage...")
    skill_endpoints = defaultdict(list)

    for skill in range(8):
        for _ in range(num_episodes):
            state, info = env.reset()
            start_pos = tuple(info['agent_pos'])

            for _ in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                if done:
                    break

            end_pos = tuple(info['agent_pos'])
            skill_endpoints[skill].append({
                'start': start_pos,
                'end': end_pos,
                'delta': (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            })

    # Plot 1: Skill movement vectors
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 8))

    for skill in range(8):
        ax = axes[skill]
        draw_env_grid(ax, env, alpha=0.3)

        for ep_data in skill_endpoints[skill]:
            start = ep_data['start']
            end = ep_data['end']

            # Draw arrow from start to end
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color=colors[skill],
                                      lw=1.5, alpha=0.6))
            ax.scatter(*start, c='green', s=50, zorder=5, marker='o')
            ax.scatter(*end, c='red', s=50, zorder=5, marker='*')

        # Calculate average movement
        deltas = [ep['delta'] for ep in skill_endpoints[skill]]
        avg_dx = np.mean([d[0] for d in deltas])
        avg_dy = np.mean([d[1] for d in deltas])

        ax.set_title(f'Skill {skill}\nAvg: ({avg_dx:.1f}, {avg_dy:.1f})')

    plt.suptitle('Skill Movement Patterns (green=start, red=end)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'skill_movements.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/skill_movements.png")

    # Plot 2: Goal-reaching simulation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Simulate reaching different goal positions (dynamically sampled from walkable cells)
    test_goals = get_valid_goals(env, num_goals=5)

    for idx, goal in enumerate(test_goals):
        ax = axes[idx]
        draw_env_grid(ax, env, alpha=0.3)

        # Mark goal
        ax.scatter(*goal, c='gold', s=300, marker='*', edgecolors='black',
                  linewidths=2, zorder=10, label='Goal')

        # For each skill, show trajectory toward goal
        state, info = env.reset()
        start = info['agent_pos']

        best_skill = None
        best_dist = float('inf')

        for skill in range(8):
            state, info = env.reset()
            positions = [tuple(info['agent_pos'])]

            for _ in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                positions.append(tuple(info['agent_pos']))
                if done:
                    break

            final_pos = positions[-1]
            dist = np.sqrt((final_pos[0]-goal[0])**2 + (final_pos[1]-goal[1])**2)

            if dist < best_dist:
                best_dist = dist
                best_skill = skill

            # Draw trajectory (faint)
            positions = np.array(positions)
            ax.plot(positions[:, 0], positions[:, 1],
                   color=colors[skill], alpha=0.3, linewidth=1)

        # Highlight best skill trajectory
        state, info = env.reset()
        positions = [tuple(info['agent_pos'])]
        for _ in range(max_steps):
            action = agent.select_action(state, best_skill, deterministic=True)
            state, _, done, _, info = env.step(action)
            positions.append(tuple(info['agent_pos']))
            if done:
                break

        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1],
               color=colors[best_skill], alpha=1, linewidth=3,
               label=f'Best: Skill {best_skill}')

        ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100,
                  marker='o', zorder=10, label='Start')

        ax.set_title(f'Goal: {goal}\nBest skill: {best_skill} (dist: {best_dist:.1f})')
        ax.legend(loc='upper right', fontsize=8)

    # Summary in last subplot
    ax = axes[5]
    ax.axis('off')
    ax.text(0.5, 0.7, 'Hierarchical Controller\nwould learn to select:',
            ha='center', va='center', fontsize=14, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.3,
            '• Skill for → when goal is right\n'
            '• Skill for ← when goal is left\n'
            '• Skill for ↑ when goal is up\n'
            '• Skill for ↓ when goal is down\n\n'
            'Then chain skills to reach any goal!',
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes)

    plt.suptitle('Which Skill Reaches Each Goal?', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'goal_skill_selection.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/goal_skill_selection.png")

    # Plot 3: Skill direction summary
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_env_grid(ax, env, alpha=0.2)

    # Draw average movement vector for each skill from center
    center = (grid_size // 2, grid_size // 2)

    for skill in range(8):
        deltas = [ep['delta'] for ep in skill_endpoints[skill]]
        avg_dx = np.mean([d[0] for d in deltas])
        avg_dy = np.mean([d[1] for d in deltas])

        # Scale for visibility
        scale = 3
        ax.annotate('',
                   xy=(center[0] + avg_dx * scale, center[1] + avg_dy * scale),
                   xytext=center,
                   arrowprops=dict(arrowstyle='->', color=colors[skill], lw=3))

        # Label
        ax.text(center[0] + avg_dx * scale * 1.2,
               center[1] + avg_dy * scale * 1.2,
               f'S{skill}', fontsize=12, fontweight='bold', color=colors[skill])

    ax.scatter(*center, c='black', s=200, zorder=10)
    ax.set_title('Skill Direction Summary\n(Average movement vector per skill)', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'skill_directions.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/skill_directions.png")

    env.close()
    print(f"\nAll hierarchical visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to DIAYN checkpoint")
    parser.add_argument("--env", default="fourrooms")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output_dir", default="plots_hierarchical")
    args = parser.parse_args()

    visualize_skill_selection(args.checkpoint, args.env, args.episodes, output_dir=args.output_dir)
