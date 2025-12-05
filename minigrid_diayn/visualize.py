"""
Visualization functions for DIAYN skills.

Creates plots and animations for presenting results to instructor.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from config import DIAYNConfig
from wrappers import make_env, get_env_info
from diayn_agent import DIAYNAgent


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_training_curves(metrics: Dict[str, List], save_path: str):
    """
    Plot training curves (4-panel figure).

    Panels:
    1. Episode pseudo-rewards over time
    2. Discriminator accuracy over time
    3. Policy entropy over time
    4. Alpha (temperature) over time

    Args:
        metrics: Dictionary of training metrics
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Smooth function for cleaner plots
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # 1. Episode rewards
    ax = axes[0, 0]
    rewards = metrics['episode_rewards']
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
    ax.plot(smooth(rewards), color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Pseudo-Reward')
    ax.set_title('Episode Pseudo-Rewards')
    ax.legend()

    # 2. Discriminator accuracy
    ax = axes[0, 1]
    if 'discriminator_accuracy' in metrics and len(metrics['discriminator_accuracy']) > 0:
        disc_acc = metrics['discriminator_accuracy']
        ax.plot(disc_acc, alpha=0.3, color='green', label='Raw')
        ax.plot(smooth(disc_acc), color='green', linewidth=2, label='Smoothed')
        ax.axhline(y=1/8, color='red', linestyle='--', label='Random (1/8)')
        ax.set_ylim([0, 1])
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Discriminator Accuracy')
    ax.legend()

    # 3. Policy entropy
    ax = axes[1, 0]
    if 'entropy' in metrics and len(metrics['entropy']) > 0:
        entropy = metrics['entropy']
        ax.plot(entropy, alpha=0.3, color='purple', label='Raw')
        ax.plot(smooth(entropy), color='purple', linewidth=2, label='Smoothed')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Policy Entropy')
    ax.legend()

    # 4. Alpha (temperature)
    ax = axes[1, 1]
    if 'alpha' in metrics and len(metrics['alpha']) > 0:
        alpha = metrics['alpha']
        ax.plot(alpha, color='orange', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Alpha')
    ax.set_title('Entropy Coefficient (Alpha)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_skill_heatmaps(
    agent: DIAYNAgent,
    env,
    num_episodes: int = 20,
    max_steps: int = 100,
    save_path: str = 'skill_heatmaps.png'
):
    """
    Plot heatmaps showing where each skill tends to visit.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes: Episodes to run per skill
        max_steps: Maximum steps per episode
        save_path: Path to save the figure
    """
    num_skills = agent.num_skills
    grid_width = env.unwrapped.width
    grid_height = env.unwrapped.height

    # Collect visit counts for each skill
    visit_counts = {}
    for skill in range(num_skills):
        counts = np.zeros((grid_height, grid_width))

        for _ in range(num_episodes):
            state, info = env.reset()

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)

                pos = info['agent_pos']
                counts[pos[1], pos[0]] += 1  # Note: y, x for array indexing

                if terminated or truncated:
                    break

        visit_counts[skill] = counts

    # Create subplot grid
    cols = 4
    rows = (num_skills + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_skills > 1 else [axes]

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('skill_cmap', ['white', 'blue', 'red'])

    for skill in range(num_skills):
        ax = axes[skill]
        counts = visit_counts[skill]

        # Normalize
        if counts.max() > 0:
            counts = counts / counts.max()

        im = ax.imshow(counts, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Skill {skill}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused subplots
    for i in range(num_skills, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Skill Visit Heatmaps', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Skill heatmaps saved to {save_path}")


def plot_skill_trajectories(
    agent: DIAYNAgent,
    env,
    num_trajectories: int = 3,
    max_steps: int = 100,
    save_path: str = 'skill_trajectories.png'
):
    """
    Plot example trajectories for each skill with arrows.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_trajectories: Trajectories to show per skill
        max_steps: Maximum steps per episode
        save_path: Path to save the figure
    """
    num_skills = agent.num_skills
    grid_width = env.unwrapped.width
    grid_height = env.unwrapped.height

    # Collect trajectories
    trajectories = defaultdict(list)

    for skill in range(num_skills):
        for _ in range(num_trajectories):
            state, info = env.reset()
            positions = [tuple(info['agent_pos'])]

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)
                positions.append(tuple(info['agent_pos']))

                if terminated or truncated:
                    break

            trajectories[skill].append(positions)

    # Create subplot grid
    cols = 4
    rows = (num_skills + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_skills > 1 else [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, num_trajectories))

    for skill in range(num_skills):
        ax = axes[skill]

        # Draw grid
        ax.set_xlim(-0.5, grid_width - 0.5)
        ax.set_ylim(-0.5, grid_height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Draw grid lines
        for x in range(grid_width + 1):
            ax.axvline(x - 0.5, color='lightgray', linewidth=0.5)
        for y in range(grid_height + 1):
            ax.axhline(y - 0.5, color='lightgray', linewidth=0.5)

        # Draw trajectories
        for traj_idx, positions in enumerate(trajectories[skill]):
            positions = np.array(positions)

            # Draw path
            ax.plot(positions[:, 0], positions[:, 1],
                   color=colors[traj_idx], linewidth=2, alpha=0.7)

            # Draw start point (circle)
            ax.scatter(positions[0, 0], positions[0, 1],
                      color=colors[traj_idx], s=100, marker='o', zorder=5)

            # Draw end point (star)
            ax.scatter(positions[-1, 0], positions[-1, 1],
                      color=colors[traj_idx], s=150, marker='*', zorder=5)

            # Draw arrows along path
            for i in range(0, len(positions) - 1, max(1, len(positions) // 5)):
                dx = positions[i + 1, 0] - positions[i, 0]
                dy = positions[i + 1, 1] - positions[i, 1]
                if dx != 0 or dy != 0:
                    ax.annotate('', xy=(positions[i + 1, 0], positions[i + 1, 1]),
                               xytext=(positions[i, 0], positions[i, 1]),
                               arrowprops=dict(arrowstyle='->', color=colors[traj_idx],
                                             lw=1.5, alpha=0.7))

        ax.set_title(f'Skill {skill}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Hide unused subplots
    for i in range(num_skills, len(axes)):
        axes[i].axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='gray', label='Start',
                  markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='*', color='gray', label='End',
                  markersize=12, linestyle='None'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Skill Trajectories', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Skill trajectories saved to {save_path}")


def plot_confusion_matrix(
    agent: DIAYNAgent,
    env,
    num_episodes: int = 10,
    max_steps: int = 100,
    save_path: str = 'confusion_matrix.png'
):
    """
    Plot discriminator confusion matrix.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes: Episodes to run per skill
        max_steps: Maximum steps per episode
        save_path: Path to save the figure
    """
    import torch

    num_skills = agent.num_skills
    confusion = np.zeros((num_skills, num_skills), dtype=np.float32)

    for true_skill in range(num_skills):
        for _ in range(num_episodes):
            state, info = env.reset()

            for step in range(max_steps):
                action = agent.select_action(state, true_skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)

                # Predict skill
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    predicted = agent.discriminator.predict_skill(state_tensor).item()

                confusion[true_skill, predicted] += 1

                if terminated or truncated:
                    break

    # Normalize rows
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_normalized = confusion / (row_sums + 1e-8)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_normalized, cmap='Blues', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Prediction Rate', rotation=270, labelpad=20)

    # Add labels
    ax.set_xticks(range(num_skills))
    ax.set_yticks(range(num_skills))
    ax.set_xticklabels([f'{i}' for i in range(num_skills)])
    ax.set_yticklabels([f'{i}' for i in range(num_skills)])
    ax.set_xlabel('Predicted Skill', fontsize=12)
    ax.set_ylabel('True Skill', fontsize=12)
    ax.set_title('Discriminator Confusion Matrix', fontsize=14)

    # Add text annotations
    for i in range(num_skills):
        for j in range(num_skills):
            value = confusion_normalized[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=color, fontsize=10)

    # Compute accuracy
    accuracy = np.trace(confusion) / confusion.sum()
    ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_skill_trajectories_with_direction(
    agent: DIAYNAgent,
    env,
    num_episodes: int = 10,
    max_steps: int = 100,
    save_path: str = 'skill_trajectories_direction.png'
):
    """
    Plot trajectories showing BOTH position AND facing direction.

    This reveals if DIAYN learned orientation skills (different directions
    at same positions) vs spatial skills (different positions).

    Direction encoding:
    - 0: Right (→)
    - 1: Down (↓)
    - 2: Left (←)
    - 3: Up (↑)

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes: Episodes to run per skill
        max_steps: Maximum steps per episode
        save_path: Path to save the figure
    """
    num_skills = agent.num_skills
    grid_width = env.unwrapped.width
    grid_height = env.unwrapped.height

    # Direction vectors for arrows
    dir_vectors = {
        0: (0.3, 0),    # Right
        1: (0, 0.3),    # Down
        2: (-0.3, 0),   # Left
        3: (0, -0.3),   # Up
    }
    dir_names = {0: '→', 1: '↓', 2: '←', 3: '↑'}

    # Collect (position, direction) data for each skill
    skill_data = defaultdict(list)

    for skill in range(num_skills):
        for _ in range(num_episodes):
            state, info = env.reset()

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)

                pos = tuple(info['agent_pos'])
                direction = info['agent_dir']
                skill_data[skill].append((pos[0], pos[1], direction))

                if terminated or truncated:
                    break

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, num_skills))

    # Plot 1: All skills with direction arrows
    ax = axes[0]
    ax.set_xlim(-0.5, grid_width - 0.5)
    ax.set_ylim(-0.5, grid_height - 0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Draw grid
    for x in range(grid_width + 1):
        ax.axvline(x - 0.5, color='lightgray', linewidth=0.5)
    for y in range(grid_height + 1):
        ax.axhline(y - 0.5, color='lightgray', linewidth=0.5)

    # Plot arrows for each skill
    for skill in range(num_skills):
        data = skill_data[skill]
        # Subsample for clarity
        sample_size = min(100, len(data))
        sampled = [data[i] for i in np.random.choice(len(data), sample_size, replace=False)]

        for x, y, d in sampled:
            dx, dy = dir_vectors[d]
            # Add small jitter to avoid overlap
            jx = np.random.uniform(-0.15, 0.15)
            jy = np.random.uniform(-0.15, 0.15)
            ax.arrow(x + jx, y + jy, dx, dy,
                    head_width=0.12, head_length=0.08,
                    fc=colors[skill], ec=colors[skill], alpha=0.6)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('All Skills: Position + Direction\n(arrows show facing direction)', fontsize=14)

    # Create legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=f'Skill {i}')
                      for i in range(num_skills)]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Plot 2: Direction distribution per skill
    ax = axes[1]

    # Count directions per skill
    dir_counts = np.zeros((num_skills, 4))
    for skill in range(num_skills):
        for x, y, d in skill_data[skill]:
            dir_counts[skill, d] += 1

    # Normalize
    dir_counts = dir_counts / (dir_counts.sum(axis=1, keepdims=True) + 1e-8)

    # Show dominant direction per skill as large arrows in a grid
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Dominant Direction per Skill', fontsize=14)

    # Arrow directions: 0=Right, 1=Down, 2=Left, 3=Up
    arrow_dx = {0: 0.3, 1: 0, 2: -0.3, 3: 0}
    arrow_dy = {0: 0, 1: -0.3, 2: 0, 3: 0.3}

    for skill in range(num_skills):
        # Position in 2x4 grid
        col = skill % 4
        row = 1 - skill // 4  # Top row first
        cx, cy = col + 0.5, row + 0.5

        dominant_dir = np.argmax(dir_counts[skill])
        pct = dir_counts[skill, dominant_dir] * 100

        # Draw box
        rect = plt.Rectangle((col + 0.05, row + 0.05), 0.9, 0.9,
                             fill=False, edgecolor=colors[skill], linewidth=2)
        ax.add_patch(rect)

        # Skill label
        ax.text(cx, cy + 0.3, f'Skill {skill}', ha='center', va='center',
               fontsize=10, fontweight='bold', color=colors[skill])

        # Big arrow showing dominant direction
        dx, dy = arrow_dx[dominant_dir], arrow_dy[dominant_dir]
        ax.annotate('', xy=(cx + dx, cy - 0.05 + dy), xytext=(cx - dx, cy - 0.05 - dy),
                   arrowprops=dict(arrowstyle='->', color=colors[skill], lw=3))

        # Direction label
        ax.text(cx, cy - 0.35, f'{dir_names[dominant_dir]} ({pct:.0f}%)',
               ha='center', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Direction-aware trajectories saved to {save_path}")

    # Print summary
    print("\n--- Direction Distribution Summary ---")
    for skill in range(num_skills):
        dominant_dir = np.argmax(dir_counts[skill])
        print(f"Skill {skill}: Dominant direction = {dir_names[dominant_dir]} "
              f"({dir_counts[skill, dominant_dir]*100:.1f}%)")


def plot_skill_diversity_scatter(
    agent: DIAYNAgent,
    env,
    num_episodes: int = 20,
    max_steps: int = 100,
    save_path: str = 'skill_diversity.png'
):
    """
    Plot final positions of each skill as a scatter plot.

    Different colors for different skills shows diversity.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        num_episodes: Episodes to run per skill
        max_steps: Maximum steps per episode
        save_path: Path to save the figure
    """
    num_skills = agent.num_skills

    # Collect final positions
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

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, num_skills))

    for skill in range(num_skills):
        positions = np.array(final_positions[skill])
        # Add jitter for visibility
        jitter = np.random.normal(0, 0.1, positions.shape)
        positions = positions + jitter

        ax.scatter(positions[:, 0], positions[:, 1],
                  c=[colors[skill]], s=100, alpha=0.6,
                  label=f'Skill {skill}')

        # Draw centroid
        centroid = np.mean(final_positions[skill], axis=0)
        ax.scatter(centroid[0], centroid[1], c=[colors[skill]],
                  s=300, marker='X', edgecolors='black', linewidths=2)

    ax.set_xlim(-0.5, env.unwrapped.width - 0.5)
    ax.set_ylim(-0.5, env.unwrapped.height - 0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Skill Final Positions (X = centroid)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Draw grid
    for x in range(env.unwrapped.width + 1):
        ax.axvline(x - 0.5, color='lightgray', linewidth=0.5)
    for y in range(env.unwrapped.height + 1):
        ax.axhline(y - 0.5, color='lightgray', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Skill diversity plot saved to {save_path}")


def create_skill_demo_gif(
    agent: DIAYNAgent,
    env,
    max_steps: int = 50,
    save_path: str = 'skill_demo.gif'
):
    """
    Create animated GIF showing all skills running.

    Args:
        agent: Trained DIAYN agent
        env: MiniGrid environment
        max_steps: Maximum steps per episode
        save_path: Path to save the GIF
    """
    if not HAS_IMAGEIO:
        print("imageio not installed. Skipping GIF creation.")
        return

    from minigrid.wrappers import RGBImgObsWrapper

    num_skills = agent.num_skills

    # We need to create a new env for rendering
    import gymnasium as gym
    render_env = gym.make(env.unwrapped.spec.id, render_mode='rgb_array')
    render_env = RGBImgObsWrapper(render_env)

    frames = []

    for skill in range(num_skills):
        obs, info = render_env.reset()
        state = env.reset()[0]  # Get flat observation from our wrapped env

        # Add skill label to frame
        for step in range(max_steps):
            # Get frame
            frame = render_env.render()

            # Add text overlay
            from PIL import Image, ImageDraw, ImageFont
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # Add skill number
            draw.text((10, 10), f"Skill {skill}", fill=(255, 255, 255))

            frames.append(np.array(img))

            # Step
            action = agent.select_action(state, skill, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)
            render_env.step(action)

            if terminated or truncated:
                # Hold on last frame
                for _ in range(10):
                    frames.append(np.array(img))
                break

    render_env.close()

    # Save GIF
    imageio.mimsave(save_path, frames, fps=10)
    print(f"Skill demo GIF saved to {save_path}")


def create_all_visualizations(
    agent: DIAYNAgent,
    config: DIAYNConfig,
    metrics: Optional[Dict] = None,
    output_dir: str = 'plots'
):
    """
    Create all visualizations for instructor presentation.

    Args:
        agent: Trained DIAYN agent
        config: Configuration
        metrics: Training metrics (optional, loads from file if None)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create environment
    env = make_env(config.env_name, fully_observable=True, seed=config.seed)

    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    # 1. Training curves
    if metrics is not None:
        plot_training_curves(
            metrics,
            os.path.join(output_dir, 'training_curves.png')
        )

    # 2. Skill heatmaps
    plot_skill_heatmaps(
        agent, env,
        num_episodes=20,
        max_steps=config.max_steps,
        save_path=os.path.join(output_dir, 'skill_heatmaps.png')
    )

    # 3. Skill trajectories
    plot_skill_trajectories(
        agent, env,
        num_trajectories=3,
        max_steps=config.max_steps,
        save_path=os.path.join(output_dir, 'skill_trajectories.png')
    )

    # 4. Confusion matrix
    plot_confusion_matrix(
        agent, env,
        num_episodes=10,
        max_steps=config.max_steps,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    # 5. Direction-aware trajectories (reveals orientation skills)
    plot_skill_trajectories_with_direction(
        agent, env,
        num_episodes=10,
        max_steps=config.max_steps,
        save_path=os.path.join(output_dir, 'skill_trajectories_direction.png')
    )

    # 6. Skill diversity scatter
    plot_skill_diversity_scatter(
        agent, env,
        num_episodes=20,
        max_steps=config.max_steps,
        save_path=os.path.join(output_dir, 'skill_diversity.png')
    )

    env.close()

    print("\n" + "=" * 60)
    print(f"All visualizations saved to {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    from train import load_metrics

    parser = argparse.ArgumentParser(description='Visualize DIAYN results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--metrics', type=str, default=None,
                        help='Path to metrics JSON file')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                        help='Environment name')
    parser.add_argument('--num_skills', type=int, default=8,
                        help='Number of skills')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Output directory for plots')

    args = parser.parse_args()

    # Load config
    from config import get_config
    config = get_config(env_name=args.env, num_skills=args.num_skills)

    # Create agent
    env = make_env(config.env_name, fully_observable=True)
    env_info = get_env_info(env)

    agent = DIAYNAgent(
        config=config,
        obs_dim=env_info['obs_dim'],
        num_actions=env_info['num_actions']
    )
    agent.load(args.checkpoint)
    env.close()

    # Load metrics if provided
    metrics = None
    if args.metrics:
        metrics = load_metrics(args.metrics)

    # Create visualizations
    create_all_visualizations(agent, config, metrics, args.output_dir)
