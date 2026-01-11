"""Visualize DIAYN training results and skill behaviors."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from core import DIAYNConfig, make_env, resolve_run
from agents import DIAYNAgent

# === Publication-Quality Visualization Settings ===

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Colorblind-friendly palette (Wong, 2011 - Nature Methods)
# These colors are distinguishable for most types of color blindness
SKILL_COLORS = [
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion/Orange
    '#009E73',  # Bluish green
    '#CC79A7',  # Reddish purple
    '#F0E442',  # Yellow
    '#56B4E9',  # Sky blue
    '#E69F00',  # Orange
    '#000000',  # Black
    '#882255',  # Wine
    '#44AA99',  # Teal
    '#332288',  # Indigo
    '#DDCC77',  # Sand
    '#117733',  # Green
    '#88CCEE',  # Cyan
    '#AA4499',  # Purple
    '#999933',  # Olive
]

# Professional color scheme
WALL_COLOR = '#404040'      # Dark gray (prints well)
WALL_EDGE = '#2A2A2A'       # Darker edge
BACKGROUND = 'white'

# Publication settings
DPI = 300
FORMATS = ['pdf', 'png']  # Both vector and raster formats

# Figure sizes (inches) - for LaTeX documents
# Single column: ~3.5", Double column: ~7"
FIG_SIZES = {
    'single': (3.5, 3.0),
    'double': (7.0, 3.0),
    'square': (3.5, 3.5),
    'grid': (7.0, 7.0),
    'wide': (7.0, 4.0),
}

# Font sizes optimized for publication (readable when scaled)
FONT_SIZES = {
    'title': 11,
    'label': 10,
    'tick': 9,
    'legend': 8,
    'annotation': 8,
}


def get_skill_colors(num_skills):
    """Get colorblind-friendly colors for skills."""
    if num_skills <= len(SKILL_COLORS):
        return SKILL_COLORS[:num_skills]
    # Fall back to a perceptually uniform colormap
    return plt.cm.viridis(np.linspace(0, 0.9, num_skills))


def save_figure(output_dir: Path, name: str):
    """Save figure in all configured formats (PDF and PNG)."""
    for fmt in FORMATS:
        plt.savefig(output_dir / f"{name}.{fmt}", dpi=DPI, facecolor=BACKGROUND)


def get_wall_positions(env):
    """Extract wall positions from MiniGrid environment."""
    grid = env.unwrapped.grid
    walls = []
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "wall":
                walls.append((x, y))
    return walls


def get_object_positions(env):
    """Extract positions of doors, keys, and goals."""
    grid = env.unwrapped.grid
    objects = {"door": [], "key": [], "goal": []}
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == "door":
                    objects["door"].append((x, y))
                elif cell.type == "key":
                    objects["key"].append((x, y))
                elif cell.type == "goal":
                    objects["goal"].append((x, y))
    return objects


def draw_walls(ax, env, alpha=0.9):
    """Draw walls on a matplotlib axis."""
    walls = get_wall_positions(env)
    for x, y in walls:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                              facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                              alpha=alpha, linewidth=1.0)
        ax.add_patch(rect)


def draw_objects(ax, env):
    """Draw doors, keys, goals on axis."""
    objects = get_object_positions(env)

    # Draw doors (brown rectangle with gap)
    for x, y in objects["door"]:
        rect = plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                              facecolor="#8B4513", edgecolor="#5D3A1A",
                              linewidth=2, zorder=10)
        ax.add_patch(rect)

    # Draw keys (yellow star marker)
    for x, y in objects["key"]:
        ax.scatter(x, y, marker='*', s=200, c='gold',
                   edgecolor='orange', linewidth=1, zorder=10)

    # Draw goals (green square)
    for x, y in objects["goal"]:
        rect = plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                              facecolor="#32CD32", edgecolor="#228B22",
                              linewidth=2, zorder=10)
        ax.add_patch(rect)


def plot_training_curves_main(metrics: dict, output_dir: Path, num_skills: int = 8):
    """Plot main training metrics: reward + discriminator accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['double'], facecolor=BACKGROUND)

    # Color scheme (colorblind-friendly)
    colors = {'reward': '#0072B2', 'reward_smooth': '#000000', 'accuracy': '#009E73'}

    # Rewards
    ax = axes[0]
    rewards = metrics["episode_rewards"]
    ax.plot(rewards, alpha=0.25, color=colors['reward'], linewidth=0.8)
    window = min(100, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, linewidth=1.5,
                color=colors['reward_smooth'], label=f'{window}-ep avg')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Intrinsic Reward")
    ax.set_title("(a) Episode Reward", fontsize=FONT_SIZES['title'])
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Discriminator accuracy
    ax = axes[1]
    if "discriminator_accuracy" in metrics and metrics["discriminator_accuracy"]:
        acc = metrics["discriminator_accuracy"]
        ax.plot(acc, color=colors['accuracy'], linewidth=1.2)
        ax.axhline(y=1/num_skills, color='gray', linestyle='--', linewidth=1, label='Random')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("(b) Discriminator Accuracy", fontsize=FONT_SIZES['title'])
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=7)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout(pad=0.5)
    save_figure(output_dir, "training_main")
    plt.close()


def plot_training_curves_aux(metrics: dict, output_dir: Path):
    """Plot auxiliary training metrics: losses + entropy."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['double'], facecolor=BACKGROUND)

    # Color scheme (colorblind-friendly)
    colors = {'disc_loss': '#D55E00', 'policy_loss': '#CC79A7', 'entropy': '#E69F00'}

    # Losses
    ax = axes[0]
    if "discriminator_loss" in metrics and metrics["discriminator_loss"]:
        ax.plot(metrics["discriminator_loss"], label="Discriminator",
                color=colors['disc_loss'], linewidth=1.2, alpha=0.9)
    if "policy_loss" in metrics and metrics["policy_loss"]:
        ax.plot(metrics["policy_loss"], label="Policy",
                color=colors['policy_loss'], linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training Losses", fontsize=FONT_SIZES['title'])
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Entropy
    ax = axes[1]
    if "entropy" in metrics and metrics["entropy"]:
        ax.plot(metrics["entropy"], color=colors['entropy'], linewidth=1.2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Policy Entropy")
        ax.set_title("(b) Policy Entropy", fontsize=FONT_SIZES['title'])
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout(pad=0.5)
    save_figure(output_dir, "training_aux")
    plt.close()


def plot_skill_trajectories(agent, env, output_dir: Path, walls: list, grid_size: int,
                            episodes_per_skill=5):
    """Plot all skill trajectories overlaid - publication quality."""
    colors = get_skill_colors(agent.num_skills)
    wall_set = set(walls)

    fig, ax = plt.subplots(figsize=FIG_SIZES['square'], facecolor=BACKGROUND)

    # Draw walls
    for x, y in walls:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                              facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                              alpha=1.0, linewidth=0.5)
        ax.add_patch(rect)

    for skill in range(agent.num_skills):
        all_positions = []
        end_positions = []

        for ep in range(episodes_per_skill):
            state, info = env.reset()
            positions = [list(info["agent_pos"])]

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                positions.append(list(info["agent_pos"]))
                if done:
                    break

            all_positions.extend(positions)
            end_positions.append(positions[-1])

        # Plot visited positions
        all_pos = np.array(all_positions)
        ax.scatter(all_pos[:, 0], all_pos[:, 1], color=colors[skill], s=12, alpha=0.6,
                   edgecolor='none', label=f"z={skill}")

        # End markers
        ends = np.array(end_positions)
        ax.scatter(ends[:, 0], ends[:, 1], color=colors[skill], s=50,
                   edgecolor='white', linewidth=1, zorder=6, marker='s')

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(0, grid_size, 4))
    ax.set_yticks(range(0, grid_size, 4))

    # Legend outside the plot (to the right)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95,
              ncol=1, handletextpad=0.3, borderpad=0.3)

    plt.tight_layout(pad=0.3)
    save_figure(output_dir, "skill_trajectories")
    plt.close()


def plot_skill_trajectories_grid(agent, env, output_dir: Path, walls: list, grid_size: int,
                                  episodes_per_skill=10):
    """Plot each skill's trajectory in separate subplot - publication quality."""
    colors = get_skill_colors(agent.num_skills)
    wall_set = set(walls)

    cols = min(4, agent.num_skills)
    rows = (agent.num_skills + cols - 1) // cols

    # Size for publication: fits in double-column width
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.7 * rows),
                             facecolor=BACKGROUND)
    axes = np.atleast_2d(axes).flatten()

    for skill in range(agent.num_skills):
        ax = axes[skill]

        # Draw walls
        for x, y in walls:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                  facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                                  linewidth=0.3, zorder=2)
            ax.add_patch(rect)

        all_positions = []
        end_positions = []

        for ep in range(episodes_per_skill):
            state, info = env.reset()
            positions = [tuple(info["agent_pos"])]

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                positions.append(tuple(info["agent_pos"]))
                if done:
                    break

            valid_positions = [p for p in positions if p not in wall_set]
            all_positions.extend(valid_positions)
            if valid_positions:
                end_positions.append(valid_positions[-1])

        # Plot positions
        if all_positions:
            all_pos = np.array(all_positions)
            ax.scatter(all_pos[:, 0], all_pos[:, 1], color=colors[skill], s=8,
                       alpha=0.7, edgecolor='none', zorder=3)

        # End markers
        if end_positions:
            ends = np.array(end_positions)
            ax.scatter(ends[:, 0], ends[:, 1], marker='s', s=25,
                       c=colors[skill], edgecolor='white', linewidth=0.5, zorder=4)

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        # Use z notation for skill (common in DIAYN literature)
        ax.set_title(f"z = {skill}", fontsize=FONT_SIZES['annotation'], pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for i in range(agent.num_skills, len(axes)):
        axes[i].axis("off")

    # Increase vertical spacing to prevent title overlap
    plt.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.3)
    save_figure(output_dir, "skill_trajectories_grid")
    plt.close()


def plot_skill_heatmaps(agent, env, output_dir: Path, walls: list, grid_size: int,
                        episodes_per_skill=10):
    """Plot visitation heatmaps for each skill - publication quality."""
    colors = get_skill_colors(agent.num_skills)
    cols = min(4, agent.num_skills)
    rows = (agent.num_skills + cols - 1) // cols

    wall_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for x, y in walls:
        wall_mask[x, y] = True

    # Publication size
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.7 * rows),
                             facecolor=BACKGROUND)
    axes = np.atleast_2d(axes).flatten()

    all_heatmaps = []

    for skill in range(agent.num_skills):
        heatmap = np.zeros((grid_size, grid_size))

        for _ in range(episodes_per_skill):
            state, info = env.reset()
            heatmap[info["agent_pos"][0], info["agent_pos"][1]] += 1

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                heatmap[info["agent_pos"][0], info["agent_pos"][1]] += 1
                if done:
                    break

        all_heatmaps.append(heatmap)

        ax = axes[skill]

        # Use consistent coordinate system (draw as rectangles)
        max_val = heatmap.max() if heatmap.max() > 0 else 1
        cmap = plt.cm.YlOrRd

        for x in range(grid_size):
            for y in range(grid_size):
                if wall_mask[x, y]:
                    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                          facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                                          linewidth=0.2)
                else:
                    intensity = heatmap[x, y] / max_val
                    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                          facecolor=cmap(intensity), edgecolor='none')
                ax.add_patch(rect)

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_title(f"z = {skill}", fontsize=FONT_SIZES['annotation'], pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    for i in range(agent.num_skills, len(axes)):
        axes[i].axis("off")

    # Increase vertical spacing to prevent title overlap
    plt.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.3)
    save_figure(output_dir, "skill_heatmaps")
    plt.close()

    return all_heatmaps, wall_mask


def plot_combined_heatmap(agent, env, output_dir: Path, walls: list, grid_size: int,
                          episodes_per_skill=10):
    """Plot combined heatmap - publication quality."""
    colors = get_skill_colors(agent.num_skills)

    wall_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for x, y in walls:
        wall_mask[x, y] = True

    # Collect heatmaps
    all_heatmaps = np.zeros((agent.num_skills, grid_size, grid_size))

    for skill in range(agent.num_skills):
        for _ in range(episodes_per_skill):
            state, info = env.reset()
            all_heatmaps[skill, info["agent_pos"][0], info["agent_pos"][1]] += 1

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                all_heatmaps[skill, info["agent_pos"][0], info["agent_pos"][1]] += 1
                if done:
                    break

    dominant_skill = np.argmax(all_heatmaps, axis=0)
    total_visits = all_heatmaps.sum(axis=0)

    # Publication size: double-column width
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZES['wide'], facecolor=BACKGROUND)

    # Left: Dominant skill
    ax = axes[0]
    for x in range(grid_size):
        for y in range(grid_size):
            if wall_mask[x, y]:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                                      linewidth=0.2)
            else:
                skill_idx = dominant_skill[x, y]
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor=colors[skill_idx], edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(0, grid_size, 4))
    ax.set_yticks(range(0, grid_size, 4))
    ax.set_title("(a) Dominant Skill", fontsize=FONT_SIZES['title'])

    # Right: Total visitation
    ax = axes[1]
    max_visits = total_visits.max() if total_visits.max() > 0 else 1
    cmap = plt.cm.viridis

    for x in range(grid_size):
        for y in range(grid_size):
            if wall_mask[x, y]:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                                      linewidth=0.2)
            else:
                intensity = total_visits[x, y] / max_visits
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor=cmap(intensity), edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(0, grid_size, 4))
    ax.set_yticks(range(0, grid_size, 4))
    ax.set_title("(b) Total Visitation", fontsize=FONT_SIZES['title'])

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_visits))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Visits')

    # Add legend for skills below the figure
    legend_elements = [Patch(facecolor=colors[i], label=f'z={i}')
                       for i in range(agent.num_skills)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=agent.num_skills,
               bbox_to_anchor=(0.35, -0.02), framealpha=0.95,
               handletextpad=0.3, columnspacing=0.8)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    save_figure(output_dir, "combined_heatmap")
    plt.close()


def plot_confusion_matrix(agent, env, output_dir: Path, episodes_per_skill=10):
    """Plot discriminator confusion matrix - publication quality."""
    confusion = np.zeros((agent.num_skills, agent.num_skills))

    for skill in range(agent.num_skills):
        for _ in range(episodes_per_skill):
            state, info = env.reset()

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)

                # Get discriminator prediction using position only
                # (position-based discriminator for better skill differentiation)
                position = info["position_normalized"]
                position_t = torch.FloatTensor(position).unsqueeze(0).to(agent.device)

                with torch.no_grad():
                    pred = agent.discriminator(position_t).argmax(dim=-1).item()

                confusion[skill, pred] += 1
                if done:
                    break

    # Normalize
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion = confusion / np.maximum(row_sums, 1)
    accuracy = np.trace(confusion) / agent.num_skills

    # Publication size: square, single-column
    fig, ax = plt.subplots(figsize=FIG_SIZES['square'], facecolor=BACKGROUND)

    # Use a perceptually uniform colormap
    sns.heatmap(
        confusion, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=range(agent.num_skills),
        yticklabels=range(agent.num_skills),
        ax=ax, annot_kws={'fontsize': 7},
        cbar_kws={'label': 'P(pred|true)', 'shrink': 0.8},
        linewidths=0.5, linecolor='white',
        vmin=0, vmax=1
    )
    ax.set_xlabel("Predicted z")
    ax.set_ylabel("True z")
    ax.set_title(f"Accuracy: {accuracy:.1%}", fontsize=FONT_SIZES['title'], pad=8)

    plt.tight_layout(pad=0.3)
    save_figure(output_dir, "confusion_matrix")
    plt.close()


def plot_tsne_embeddings(agent, env, output_dir: Path, episodes_per_skill=5):
    """Plot t-SNE of encoder representations colored by skill."""
    from sklearn.manifold import TSNE

    colors = get_skill_colors(agent.num_skills)
    embeddings = []
    skill_labels = []

    # Collect embeddings from all skills
    for skill in range(agent.num_skills):
        for _ in range(episodes_per_skill):
            state, info = env.reset()
            for step in range(100):
                # Get encoder embedding
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    emb = agent.encoder(state_t).cpu().numpy()
                embeddings.append(emb[0])
                skill_labels.append(skill)

                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                if done:
                    break

    # Run t-SNE
    embeddings = np.array(embeddings)
    skill_labels = np.array(skill_labels)

    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) // 4),
                random_state=42, max_iter=1000)
    proj = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=FIG_SIZES['square'], facecolor=BACKGROUND)

    for skill in range(agent.num_skills):
        mask = skill_labels == skill
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors[skill]],
                   s=10, alpha=0.6, label=f'z={skill}', edgecolor='none')

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95,
              fontsize=FONT_SIZES['legend'], handletextpad=0.3)

    # Remove axis ticks (t-SNE dimensions are arbitrary)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout(pad=0.3)
    save_figure(output_dir, "tsne_embeddings")
    plt.close()


def plot_skill_quiver(agent, env, output_dir: Path, walls: list, grid_size: int,
                      episodes_per_skill=5):
    """Plot skill trajectories with direction arrows."""
    colors = get_skill_colors(agent.num_skills)

    fig, ax = plt.subplots(figsize=FIG_SIZES['square'], facecolor=BACKGROUND)

    # Draw walls
    for x, y in walls:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                              facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                              alpha=1.0, linewidth=0.5)
        ax.add_patch(rect)

    for skill in range(agent.num_skills):
        all_arrows = []  # List of (x, y, dx, dy)

        for ep in range(episodes_per_skill):
            state, info = env.reset()
            positions = [list(info["agent_pos"])]

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                positions.append(list(info["agent_pos"]))
                if done:
                    break

            # Create arrows from position pairs
            for i in range(len(positions) - 1):
                x, y = positions[i]
                dx = positions[i + 1][0] - x
                dy = positions[i + 1][1] - y
                if dx != 0 or dy != 0:  # Only add if there's movement
                    all_arrows.append((x, y, dx, dy))

        # Draw arrows with quiver
        if all_arrows:
            arrows = np.array(all_arrows)
            ax.quiver(arrows[:, 0], arrows[:, 1], arrows[:, 2], arrows[:, 3],
                      color=colors[skill], alpha=0.7, scale=1, scale_units='xy',
                      angles='xy', width=0.015, headwidth=3, headlength=4,
                      label=f'z={skill}')

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(0, grid_size, 4))
    ax.set_yticks(range(0, grid_size, 4))

    # Create legend manually with patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'z={i}')
                       for i in range(agent.num_skills)]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95, fontsize=FONT_SIZES['legend'], handletextpad=0.3)

    plt.tight_layout(pad=0.3)
    save_figure(output_dir, "skill_quiver")
    plt.close()


def plot_discriminator_confidence(agent, env, output_dir: Path, walls: list,
                                   grid_size: int, episodes_per_skill=5):
    """Plot heatmap of discriminator confidence (lower entropy = more confident)."""
    import torch.nn.functional as F

    wall_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for x, y in walls:
        wall_mask[x, y] = True

    entropy_map = np.zeros((grid_size, grid_size))
    visit_count = np.zeros((grid_size, grid_size))

    for skill in range(agent.num_skills):
        for _ in range(episodes_per_skill):
            state, info = env.reset()

            for _ in range(100):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)

                # Get discriminator softmax output using position only
                # (position-based discriminator for better skill differentiation)
                position = info["position_normalized"]
                position_t = torch.FloatTensor(position).unsqueeze(0).to(agent.device)

                with torch.no_grad():
                    logits = agent.discriminator(position_t)
                    probs = F.softmax(logits, dim=-1)
                    # Entropy: -sum(p * log(p))
                    entropy = -(probs * (probs + 1e-8).log()).sum().item()

                x, y = info["agent_pos"]
                entropy_map[x, y] += entropy
                visit_count[x, y] += 1

                if done:
                    break

    # Average entropy per cell
    avg_entropy = np.divide(entropy_map, visit_count,
                            out=np.zeros_like(entropy_map),
                            where=visit_count > 0)

    # Max entropy for normalization (uniform distribution)
    max_entropy = np.log(agent.num_skills)

    # Plot
    fig, ax = plt.subplots(figsize=FIG_SIZES['square'], facecolor=BACKGROUND)

    # Use inverted colormap: dark = confident (low entropy), light = uncertain
    cmap = plt.cm.RdYlGn  # Red (uncertain) to Green (confident)

    for x in range(grid_size):
        for y in range(grid_size):
            if wall_mask[x, y]:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                                      linewidth=0.2)
            elif visit_count[x, y] > 0:
                # Normalize: 0 = max entropy (uncertain), 1 = 0 entropy (confident)
                confidence = 1 - (avg_entropy[x, y] / max_entropy)
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor=cmap(confidence), edgecolor='none')
            else:
                # Unvisited cells
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                      facecolor='#E0E0E0', edgecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(range(0, grid_size, 4))
    ax.set_yticks(range(0, grid_size, 4))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Confidence')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    plt.tight_layout(pad=0.3)
    save_figure(output_dir, "discriminator_confidence")
    plt.close()


def plot_multi_env_comparison(run_dirs: list, output_dir: Path, episodes_per_skill=5):
    """Plot skill trajectories as smooth paths with start/end markers.

    Shows actual navigation paths rather than just visited positions,
    making skill behavior patterns more visually apparent.

    Args:
        run_dirs: List of (run_dir, env_label) tuples
        output_dir: Where to save the figure
        episodes_per_skill: Episodes to run per skill
    """
    from matplotlib.patches import Patch, Circle
    from matplotlib.collections import LineCollection
    import matplotlib.gridspec as gridspec

    n_envs = len(run_dirs)
    if n_envs == 0:
        return

    # Load first to get num_skills
    first_config = DIAYNConfig.load(run_dirs[0][0] / "config.json")
    num_skills = first_config.num_skills
    colors = get_skill_colors(num_skills)

    # Create figure with 1x3 horizontal layout (better for comparing)
    fig, axes = plt.subplots(1, n_envs, figsize=(3.2 * n_envs, 3.5), facecolor=BACKGROUND)
    if n_envs == 1:
        axes = [axes]

    for idx, (run_dir, env_label) in enumerate(run_dirs[:len(axes)]):
        ax = axes[idx]

        # Load config, agent, env
        config = DIAYNConfig.load(run_dir / "config.json")
        diayn_path = run_dir / "diayn" / "final_model.pt"
        env, _ = make_env(config.env_key)
        agent = DIAYNAgent.from_checkpoint(diayn_path, config)
        grid_size = env.unwrapped.width

        # Get walls
        env.reset()
        walls = get_wall_positions(env)

        # Draw walls with nice styling
        for x, y in walls:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                  facecolor=WALL_COLOR, edgecolor=WALL_EDGE,
                                  alpha=1.0, linewidth=0.3)
            ax.add_patch(rect)

        # Draw doors (so they're visible as passable)
        objects = get_object_positions(env)
        for x, y in objects.get("door", []):
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                  facecolor='#8B4513', edgecolor='#5D2E0C',
                                  alpha=0.7, linewidth=0.5)
            ax.add_patch(rect)

        # Create wall set for filtering invalid positions
        wall_set = set(walls)

        # Collect trajectories for each skill
        for skill in range(agent.num_skills):
            trajectories = []  # List of valid trajectory arrays
            end_positions = []

            for ep in range(episodes_per_skill):
                state, info = env.reset()
                positions = [tuple(info["agent_pos"])]

                for _ in range(100):
                    action = agent.select_action(state, skill, deterministic=True)
                    state, _, done, _, info = env.step(action)
                    positions.append(tuple(info["agent_pos"]))
                    if done:
                        break

                # Keep only valid positions, but track if trajectory is valid
                valid_positions = [p for p in positions if p not in wall_set]
                if len(valid_positions) >= 1:
                    trajectories.append(np.array(valid_positions))
                    end_positions.append(valid_positions[-1])

            # Draw each trajectory with its end marker together
            for traj, end_pos in zip(trajectories, end_positions):
                # Apply jitter to entire trajectory
                jittered = []
                for x, y in traj:
                    jx = x + np.random.uniform(-0.12, 0.12)
                    jy = y + np.random.uniform(-0.12, 0.12)
                    if (round(jx), round(jy)) in wall_set:
                        jittered.append([x, y])
                    else:
                        jittered.append([jx, jy])
                jittered = np.array(jittered)

                # Draw line segments only between adjacent positions
                for i in range(len(traj) - 1):
                    p1, p2 = traj[i], traj[i + 1]
                    # Only draw if positions are adjacent (no wall between)
                    if abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) <= 1:
                        ax.plot([jittered[i, 0], jittered[i+1, 0]],
                                [jittered[i, 1], jittered[i+1, 1]],
                                color=colors[skill], alpha=0.6, linewidth=1.5,
                                zorder=2, solid_capstyle='round')

                # Draw end marker at the last point of the jittered trajectory
                ax.scatter(jittered[-1, 0], jittered[-1, 1], color=colors[skill],
                           s=80, marker='s', edgecolor='white', linewidth=1.2,
                           alpha=0.95, zorder=5)

        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_title(env_label, fontsize=FONT_SIZES['title'], fontweight='medium', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Clean border
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#CCCCCC')

        env.close()

    # Single legend at bottom
    legend_elements = [Patch(facecolor=colors[i], label=f'$z_{i}$')
                       for i in range(num_skills)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=num_skills,
               bbox_to_anchor=(0.5, 0.01), framealpha=0.95, fontsize=8,
               handletextpad=0.3, columnspacing=0.8, edgecolor='none')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Room for legend
    save_figure(output_dir, "multi_env_comparison")
    plt.close()


def visualize(run_dir: Path, episodes_per_skill: int = 10):
    config = DIAYNConfig.load(run_dir / "config.json")
    diayn_dir = run_dir / "diayn"
    diayn_path = diayn_dir / "final_model.pt"
    metrics_path = diayn_dir / "metrics.json"

    # Output to plots/ inside diayn dir
    output_dir = diayn_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Visualizing: {run_dir.name}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load metrics
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("Plotting training curves (main)...")
        plot_training_curves_main(metrics, output_dir, config.num_skills)
        print("Plotting training curves (aux)...")
        plot_training_curves_aux(metrics, output_dir)

    # Load agent and environment
    env, _ = make_env(config.env_key)
    agent = DIAYNAgent.from_checkpoint(diayn_path, config)
    grid_size = env.unwrapped.width

    # CRITICAL: Capture wall positions ONCE and reuse for ALL plots
    # This ensures consistent wall/doorway positions across all visualizations
    env.reset()
    walls = get_wall_positions(env)
    print(f"Captured {len(walls)} wall positions (fixed for all plots)")

    print(f"Plotting skill trajectories ({episodes_per_skill} episodes per skill)...")
    plot_skill_trajectories(agent, env, output_dir, walls, grid_size, episodes_per_skill)

    print("Plotting skill trajectories (grid)...")
    plot_skill_trajectories_grid(agent, env, output_dir, walls, grid_size, episodes_per_skill)

    print("Plotting skill quiver (arrows)...")
    plot_skill_quiver(agent, env, output_dir, walls, grid_size, episodes_per_skill)

    print("Plotting skill heatmaps...")
    plot_skill_heatmaps(agent, env, output_dir, walls, grid_size, episodes_per_skill)

    print("Plotting combined heatmap...")
    plot_combined_heatmap(agent, env, output_dir, walls, grid_size, episodes_per_skill)

    print("Plotting discriminator confidence...")
    plot_discriminator_confidence(agent, env, output_dir, walls, grid_size, episodes_per_skill)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(agent, env, output_dir, episodes_per_skill)

    print("Plotting t-SNE embeddings...")
    plot_tsne_embeddings(agent, env, output_dir, episodes_per_skill)

    env.close()

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="latest", help="Run dir or 'latest'")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per skill for visualization")
    parser.add_argument("--compare", nargs="+", help="Compare multiple runs: --compare run1 run2 run3")
    args = parser.parse_args()

    if args.compare:
        # Multi-environment comparison mode
        from pathlib import Path
        runs_dir = Path("runs")
        run_dirs = []

        for run_name in args.compare:
            run_path = runs_dir / run_name
            if run_path.exists():
                # Extract env name from run directory name
                env_label = run_name.split("_")[0].replace("-", " ").title()
                run_dirs.append((run_path, env_label))
            else:
                print(f"Warning: Run not found: {run_name}")

        if run_dirs:
            output_dir = Path("runs") / "comparison_plots"
            output_dir.mkdir(exist_ok=True)
            print(f"Generating multi-environment comparison...")
            print(f"Runs: {[r[1] for r in run_dirs]}")
            plot_multi_env_comparison(run_dirs, output_dir, args.episodes)
            print(f"Saved to: {output_dir}")
    else:
        run_dir = resolve_run(args.run)
        visualize(run_dir, episodes_per_skill=args.episodes)
