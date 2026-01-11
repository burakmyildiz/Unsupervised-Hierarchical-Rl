#!/usr/bin/env python3
"""Generate publication-quality comparison figures for DIAYN skill analysis.

This script creates clean, LaTeX-compatible figures comparing skill behaviors
under different training conditions (e.g., with/without movement restriction).

Usage:
    # Generate quiver comparison for two runs
    python scripts/generate_comparison_figures.py \
        --run1 runs/fourrooms_old_run \
        --run2 runs/fourrooms_20260111_173824 \
        --label1 "All Actions" \
        --label2 "Movement Only" \
        --output tex/figures/action_comparison_quiver.pdf

    # Generate single run quiver plot
    python scripts/generate_comparison_figures.py \
        --run1 runs/fourrooms_20260111_173824 \
        --output tex/figures/skill_quiver.pdf \
        --type quiver

    # Generate coverage heatmap comparison
    python scripts/generate_comparison_figures.py \
        --run1 runs/old_run --run2 runs/new_run \
        --type heatmap \
        --output tex/figures/coverage_comparison.pdf
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

# Use LaTeX-compatible settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'text.usetex': False,  # Set True if LaTeX is available
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

# Color scheme for skills (colorblind-friendly)
SKILL_COLORS = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
    '#000000',  # black
]

WALL_COLOR = '#404040'
FLOOR_COLOR = '#FFFFFF'
GRID_COLOR = '#CCCCCC'


def get_skill_colors(num_skills):
    """Get colors for skills, extending if needed."""
    if num_skills <= len(SKILL_COLORS):
        return SKILL_COLORS[:num_skills]
    # Generate more colors if needed
    cmap = plt.cm.get_cmap('tab20')
    return [cmap(i / num_skills) for i in range(num_skills)]


def load_agent_and_env(run_dir):
    """Load agent and environment from a run directory."""
    from core import DIAYNConfig, make_env
    from agents import DIAYNAgent

    run_dir = Path(run_dir)
    config = DIAYNConfig.load(run_dir / "config.json")

    env, _ = make_env(
        config.env_key,
        random_start=config.random_start,
        partial_obs=config.partial_obs,
        movement_only=getattr(config, 'movement_only', False)
    )

    model_path = run_dir / "diayn" / "final_model.pt"
    agent = DIAYNAgent.from_checkpoint(model_path, config)

    return agent, env, config


def get_wall_positions(env):
    """Get wall positions from environment."""
    walls = []
    grid = env.unwrapped.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == 'wall':
                walls.append((x, y))
    return walls


def collect_trajectories(agent, env, episodes_per_skill=10, max_steps=100):
    """Collect trajectory data for all skills.

    Returns:
        dict: {skill_idx: list of (x, y, dx, dy) arrows}
    """
    skill_arrows = {z: [] for z in range(agent.num_skills)}

    for skill in range(agent.num_skills):
        for ep in range(episodes_per_skill):
            state, info = env.reset()
            positions = [tuple(info["agent_pos"])]

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                positions.append(tuple(info["agent_pos"]))
                if done:
                    break

            # Create arrows from position pairs
            for i in range(len(positions) - 1):
                x, y = positions[i]
                dx = positions[i + 1][0] - x
                dy = positions[i + 1][1] - y
                if dx != 0 or dy != 0:  # Only if there's movement
                    skill_arrows[skill].append((x, y, dx, dy))

    return skill_arrows


def collect_visits(agent, env, episodes_per_skill=10, max_steps=100):
    """Collect visitation data for coverage analysis.

    Returns:
        tuple: (total_visits array, skill_visits dict, skill_endpoints dict)
    """
    grid_size = env.unwrapped.grid.width
    total_visits = np.zeros((grid_size, grid_size))
    skill_visits = {z: np.zeros((grid_size, grid_size)) for z in range(agent.num_skills)}
    skill_endpoints = {z: [] for z in range(agent.num_skills)}

    for skill in range(agent.num_skills):
        for ep in range(episodes_per_skill):
            state, info = env.reset()

            for step in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                x, y = info["agent_pos"]
                total_visits[x, y] += 1
                skill_visits[skill][x, y] += 1
                if done:
                    break

            # Record endpoint
            skill_endpoints[skill].append(tuple(info["agent_pos"]))

    return total_visits, skill_visits, skill_endpoints


def draw_grid_background(ax, grid_size, walls):
    """Draw the grid background with walls."""
    # Draw floor
    ax.set_facecolor(FLOOR_COLOR)

    # Draw walls
    for x, y in walls:
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                              facecolor=WALL_COLOR, edgecolor='none')
        ax.add_patch(rect)

    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color=GRID_COLOR, linewidth=0.3, zorder=0)
        ax.axvline(i - 0.5, color=GRID_COLOR, linewidth=0.3, zorder=0)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')


def plot_quiver_single(ax, skill_arrows, walls, grid_size, num_skills,
                       show_legend=True, alpha=0.7):
    """Plot quiver arrows for all skills on a single axis."""
    colors = get_skill_colors(num_skills)
    draw_grid_background(ax, grid_size, walls)

    for skill in range(num_skills):
        arrows = skill_arrows[skill]
        if arrows:
            arr = np.array(arrows)
            ax.quiver(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                      color=colors[skill], alpha=alpha,
                      scale=1, scale_units='xy', angles='xy',
                      width=0.012, headwidth=3, headlength=3.5,
                      zorder=10 + skill)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if show_legend:
        patches = [mpatches.Patch(color=colors[i], label=f'$z_{i}$')
                   for i in range(num_skills)]
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1),
                  framealpha=0.9, edgecolor='none')


def plot_heatmap_single(ax, total_visits, skill_visits, walls, grid_size,
                        num_skills, mode='dominant'):
    """Plot heatmap on a single axis.

    Args:
        mode: 'dominant' for dominant skill coloring, 'visits' for visit counts
    """
    colors = get_skill_colors(num_skills)
    draw_grid_background(ax, grid_size, walls)

    if mode == 'dominant':
        # Color each cell by dominant skill
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in walls:
                    continue
                visits = [skill_visits[z][x, y] for z in range(num_skills)]
                if sum(visits) > 0:
                    dominant = np.argmax(visits)
                    rect = plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                                          facecolor=colors[dominant],
                                          alpha=0.7, edgecolor='none')
                    ax.add_patch(rect)
    else:
        # Visit count heatmap
        masked = np.ma.masked_where(total_visits == 0, total_visits)
        im = ax.imshow(masked.T, origin='lower', cmap='YlOrRd',
                       extent=[-0.5, grid_size-0.5, -0.5, grid_size-0.5])
        return im

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def generate_quiver_comparison(run1, run2, label1, label2, output_path,
                               episodes=10, figsize=(7, 3.5)):
    """Generate side-by-side quiver comparison figure."""
    print(f"Loading run 1: {run1}")
    agent1, env1, config1 = load_agent_and_env(run1)
    env1.reset()
    walls1 = get_wall_positions(env1)
    grid_size1 = env1.unwrapped.grid.width
    arrows1 = collect_trajectories(agent1, env1, episodes_per_skill=episodes)
    env1.close()

    print(f"Loading run 2: {run2}")
    agent2, env2, config2 = load_agent_and_env(run2)
    env2.reset()
    walls2 = get_wall_positions(env2)
    grid_size2 = env2.unwrapped.grid.width
    arrows2 = collect_trajectories(agent2, env2, episodes_per_skill=episodes)
    env2.close()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_quiver_single(ax1, arrows1, walls1, grid_size1, agent1.num_skills,
                       show_legend=False)
    ax1.set_title(f'({chr(97)}) {label1}')

    plot_quiver_single(ax2, arrows2, walls2, grid_size2, agent2.num_skills,
                       show_legend=True)
    ax2.set_title(f'({chr(98)}) {label2}')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def generate_heatmap_comparison(run1, run2, label1, label2, output_path,
                                episodes=10, figsize=(7, 3.5)):
    """Generate side-by-side dominant skill heatmap comparison."""
    print(f"Loading run 1: {run1}")
    agent1, env1, config1 = load_agent_and_env(run1)
    env1.reset()
    walls1 = get_wall_positions(env1)
    grid_size1 = env1.unwrapped.grid.width
    visits1, skill_visits1, _ = collect_visits(agent1, env1, episodes_per_skill=episodes)
    env1.close()

    print(f"Loading run 2: {run2}")
    agent2, env2, config2 = load_agent_and_env(run2)
    env2.reset()
    walls2 = get_wall_positions(env2)
    grid_size2 = env2.unwrapped.grid.width
    visits2, skill_visits2, _ = collect_visits(agent2, env2, episodes_per_skill=episodes)
    env2.close()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_heatmap_single(ax1, visits1, skill_visits1, walls1, grid_size1,
                        agent1.num_skills)
    ax1.set_title(f'({chr(97)}) {label1}')

    plot_heatmap_single(ax2, visits2, skill_visits2, walls2, grid_size2,
                        agent2.num_skills)
    ax2.set_title(f'({chr(98)}) {label2}')

    # Add shared legend
    colors = get_skill_colors(agent1.num_skills)
    patches = [mpatches.Patch(color=colors[i], label=f'$z_{i}$')
               for i in range(agent1.num_skills)]
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.12, 0.5),
               framealpha=0.9, edgecolor='none')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def generate_single_quiver(run, output_path, episodes=10, figsize=(4, 4)):
    """Generate quiver plot for a single run."""
    print(f"Loading run: {run}")
    agent, env, config = load_agent_and_env(run)
    env.reset()
    walls = get_wall_positions(env)
    grid_size = env.unwrapped.grid.width
    arrows = collect_trajectories(agent, env, episodes_per_skill=episodes)
    env.close()

    fig, ax = plt.subplots(figsize=figsize)
    plot_quiver_single(ax, arrows, walls, grid_size, agent.num_skills)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


def compute_coverage_metrics(run):
    """Compute coverage metrics for a run."""
    agent, env, config = load_agent_and_env(run)
    env.reset()
    walls = set(get_wall_positions(env))
    grid_size = env.unwrapped.grid.width

    # Get navigable cells
    navigable = set()
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) not in walls:
                navigable.add((x, y))

    visits, skill_visits, _ = collect_visits(agent, env, episodes_per_skill=10)
    env.close()

    # Cell coverage
    visited = set()
    for x in range(grid_size):
        for y in range(grid_size):
            if visits[x, y] > 0:
                visited.add((x, y))

    cell_coverage = len(visited & navigable) / len(navigable)

    # Uniformity
    flat_visits = visits[visits > 0]
    if len(flat_visits) > 0:
        probs = flat_visits / flat_visits.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(navigable))
        uniformity = entropy / max_entropy
    else:
        uniformity = 0.0

    return {
        'cell_coverage': cell_coverage,
        'uniformity': uniformity,
        'cells_visited': len(visited & navigable),
        'total_navigable': len(navigable),
    }


def print_metrics_table(run1, run2, label1, label2):
    """Print LaTeX-formatted metrics comparison table."""
    print(f"\nComputing metrics for {label1}...")
    m1 = compute_coverage_metrics(run1)
    print(f"Computing metrics for {label2}...")
    m2 = compute_coverage_metrics(run2)

    print("\n" + "="*60)
    print("LaTeX Table Code:")
    print("="*60)
    print(r"""
\begin{table}[h]
\centering
\caption{Coverage Metrics Comparison}
\label{tab:coverage_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{%s} & \textbf{%s} \\
\midrule
Cell Coverage & %.1f\%% & %.1f\%% \\
Coverage Uniformity & %.2f & %.2f \\
Cells Visited & %d / %d & %d / %d \\
\bottomrule
\end{tabular}
\end{table}
""" % (label1, label2,
       m1['cell_coverage']*100, m2['cell_coverage']*100,
       m1['uniformity'], m2['uniformity'],
       m1['cells_visited'], m1['total_navigable'],
       m2['cells_visited'], m2['total_navigable']))
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison figures")
    parser.add_argument("--run1", type=str, required=True, help="First run directory")
    parser.add_argument("--run2", type=str, default=None, help="Second run directory (for comparison)")
    parser.add_argument("--label1", type=str, default="Run 1", help="Label for first run")
    parser.add_argument("--label2", type=str, default="Run 2", help="Label for second run")
    parser.add_argument("--output", type=str, default="comparison.pdf", help="Output path")
    parser.add_argument("--type", type=str, default="quiver",
                        choices=["quiver", "heatmap", "metrics"],
                        help="Type of figure to generate")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per skill")
    parser.add_argument("--width", type=float, default=7.0, help="Figure width in inches")
    parser.add_argument("--height", type=float, default=3.5, help="Figure height in inches")

    args = parser.parse_args()

    if args.type == "metrics":
        if args.run2 is None:
            print("Error: --run2 required for metrics comparison")
            sys.exit(1)
        print_metrics_table(args.run1, args.run2, args.label1, args.label2)
    elif args.run2 is None:
        # Single run
        generate_single_quiver(args.run1, args.output, episodes=args.episodes,
                               figsize=(args.width, args.height))
    else:
        # Comparison
        if args.type == "quiver":
            generate_quiver_comparison(args.run1, args.run2, args.label1, args.label2,
                                       args.output, episodes=args.episodes,
                                       figsize=(args.width, args.height))
        elif args.type == "heatmap":
            generate_heatmap_comparison(args.run1, args.run2, args.label1, args.label2,
                                        args.output, episodes=args.episodes,
                                        figsize=(args.width, args.height))
