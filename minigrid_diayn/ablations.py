"""Ablation study configurations for DIAYN experiments."""

from run_experiments import ExperimentConfig, run_experiment
from typing import List
import os


# Ablation configurations
ABLATIONS = {
    # Skill count ablation
    "skills_4": ExperimentConfig(name="skills_4", num_skills=4, seeds=[42, 123, 456]),
    "skills_8": ExperimentConfig(name="skills_8", num_skills=8, seeds=[42, 123, 456]),
    "skills_16": ExperimentConfig(name="skills_16", num_skills=16, seeds=[42, 123, 456]),
    "skills_32": ExperimentConfig(name="skills_32", num_skills=32, seeds=[42, 123, 456]),

    # Alpha ablation
    "alpha_0.05": ExperimentConfig(name="alpha_0.05", alpha=0.05, seeds=[42, 123, 456]),
    "alpha_0.1": ExperimentConfig(name="alpha_0.1", alpha=0.1, seeds=[42, 123, 456]),
    "alpha_0.2": ExperimentConfig(name="alpha_0.2", alpha=0.2, seeds=[42, 123, 456]),
    "alpha_0.5": ExperimentConfig(name="alpha_0.5", alpha=0.5, seeds=[42, 123, 456]),

    # Environment ablation
    "env_6x6": ExperimentConfig(name="env_6x6", env_key="empty-6x6", seeds=[42, 123, 456]),
    "env_8x8": ExperimentConfig(name="env_8x8", env_key="empty-8x8", seeds=[42, 123, 456]),
    "env_16x16": ExperimentConfig(name="env_16x16", env_key="empty-16x16", seeds=[42, 123, 456]),
    "env_fourrooms": ExperimentConfig(name="env_fourrooms", env_key="fourrooms", seeds=[42, 123, 456]),
}


def run_ablation_suite(ablation_names: List[str] = None, output_base: str = "experiments/ablations"):
    """Run a set of ablation experiments."""
    if ablation_names is None:
        ablation_names = list(ABLATIONS.keys())

    os.makedirs(output_base, exist_ok=True)

    results = {}
    for name in ablation_names:
        if name not in ABLATIONS:
            print(f"Unknown ablation: {name}")
            continue

        config = ABLATIONS[name]
        output_dir = os.path.join(output_base, name)

        print(f"\n{'#'*60}")
        print(f"# Ablation: {name}")
        print(f"{'#'*60}")

        results[name] = run_experiment(config, output_dir)

    return results


def run_skill_ablation():
    """Run skill count ablation."""
    return run_ablation_suite(["skills_4", "skills_8", "skills_16", "skills_32"])


def run_alpha_ablation():
    """Run entropy coefficient ablation."""
    return run_ablation_suite(["alpha_0.05", "alpha_0.1", "alpha_0.2", "alpha_0.5"])


def run_env_ablation():
    """Run environment size ablation."""
    return run_ablation_suite(["env_6x6", "env_8x8", "env_16x16"])


def compare_ablations(ablation_names: List[str], metric_key: str = "episode_rewards",
                     output_path: str = "experiments/ablation_comparison.png"):
    """Plot comparison across ablations."""
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ablation_names)))

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    for idx, name in enumerate(ablation_names):
        metrics_path = f"experiments/ablations/{name}/aggregated_metrics.json"
        if not os.path.exists(metrics_path):
            print(f"Missing: {metrics_path}")
            continue

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        if metric_key not in metrics:
            continue

        mean = np.array(metrics[metric_key]['mean'])
        std = np.array(metrics[metric_key]['std'])

        mean_smooth = smooth(mean)
        std_smooth = smooth(std)
        x = np.arange(len(mean_smooth))

        ax.plot(x, mean_smooth, color=colors[idx], linewidth=2, label=name)
        ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                       alpha=0.2, color=colors[idx])

    ax.set_xlabel('Episode')
    ax.set_ylabel(metric_key.replace('_', ' ').title())
    ax.set_title(f'Ablation Comparison: {metric_key}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["skills", "alpha", "env", "all"], default="skills")
    args = parser.parse_args()

    if args.type == "skills":
        run_skill_ablation()
    elif args.type == "alpha":
        run_alpha_ablation()
    elif args.type == "env":
        run_env_ablation()
    elif args.type == "all":
        run_skill_ablation()
        run_alpha_ablation()
        run_env_ablation()
