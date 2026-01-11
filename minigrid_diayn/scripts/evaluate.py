"""Evaluate DIAYN skills: diversity, consistency, discriminator accuracy."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core import DIAYNConfig, make_env, resolve_run
from agents import DIAYNAgent


def evaluate_diversity(agent, env, episodes_per_skill=10, max_steps=100):
    """Measure state coverage and final position spread across skills."""
    all_positions = defaultdict(list)
    final_positions = defaultdict(list)

    for skill in range(agent.num_skills):
        for _ in range(episodes_per_skill):
            state, info = env.reset()
            positions = [tuple(info["agent_pos"])]

            for _ in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                positions.append(tuple(info["agent_pos"]))
                if done:
                    break

            all_positions[skill].extend(positions)
            final_positions[skill].append(positions[-1])

    # State coverage
    unique_states = set()
    for positions in all_positions.values():
        unique_states.update(positions)
    grid_size = env.unwrapped.width * env.unwrapped.height
    coverage = len(unique_states) / grid_size

    # Final position spread
    all_finals = [p for finals in final_positions.values() for p in finals]
    if len(all_finals) > 1:
        finals_arr = np.array(all_finals)
        spread = np.std(finals_arr, axis=0).mean()
    else:
        spread = 0.0

    return {"coverage": coverage, "final_spread": spread}


def evaluate_consistency(agent, env, episodes_per_skill=20, max_steps=100):
    """Measure trajectory variance within each skill (lower = more consistent)."""
    skill_variances = []

    for skill in range(agent.num_skills):
        trajectories = []
        for _ in range(episodes_per_skill):
            state, info = env.reset()
            traj = [tuple(info["agent_pos"])]

            for _ in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)
                traj.append(tuple(info["agent_pos"]))
                if done:
                    break

            trajectories.append(np.array(traj))

        # Compute variance of final positions
        finals = np.array([t[-1] for t in trajectories])
        skill_variances.append(np.var(finals, axis=0).mean())

    return {"mean_variance": np.mean(skill_variances), "per_skill": skill_variances}


def evaluate_discriminator(agent, env, episodes_per_skill=10, max_steps=100):
    """Measure discriminator accuracy per skill."""
    import torch

    correct = defaultdict(int)
    total = defaultdict(int)

    for skill in range(agent.num_skills):
        for _ in range(episodes_per_skill):
            state, info = env.reset()

            for _ in range(max_steps):
                action = agent.select_action(state, skill, deterministic=True)
                state, _, done, _, info = env.step(action)

                # Get discriminator prediction using position only
                # (position-based discriminator for better skill differentiation)
                position = info["position_normalized"]
                position_t = torch.FloatTensor(position).unsqueeze(0).to(agent.device)

                with torch.no_grad():
                    pred = agent.discriminator(position_t).argmax(dim=-1).item()

                total[skill] += 1
                if pred == skill:
                    correct[skill] += 1

                if done:
                    break

    per_skill = {s: correct[s] / total[s] if total[s] > 0 else 0 for s in range(agent.num_skills)}
    overall = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0

    return {"overall": overall, "per_skill": per_skill}


def evaluate(run_dir: Path, episodes_per_skill: int = 10):
    config = DIAYNConfig.load(run_dir / "config.json")
    diayn_path = run_dir / "diayn" / "final_model.pt"

    env, _ = make_env(config.env_key)
    agent = DIAYNAgent.from_checkpoint(diayn_path, config)

    print("=" * 60)
    print(f"Evaluating: {run_dir.name}")
    print(f"Environment: {config.env_key}")
    print(f"Skills: {config.num_skills}")
    print("=" * 60)

    print("\nDiversity...")
    diversity = evaluate_diversity(agent, env, episodes_per_skill)
    print(f"  Coverage: {diversity['coverage']:.1%}")
    print(f"  Final spread: {diversity['final_spread']:.2f}")

    print("\nConsistency...")
    consistency = evaluate_consistency(agent, env, episodes_per_skill * 2)
    print(f"  Mean variance: {consistency['mean_variance']:.2f}")

    print("\nDiscriminator accuracy...")
    disc = evaluate_discriminator(agent, env, episodes_per_skill)
    print(f"  Overall: {disc['overall']:.1%}")
    for s, acc in disc["per_skill"].items():
        print(f"  Skill {s}: {acc:.1%}")

    env.close()

    return {"diversity": diversity, "consistency": consistency, "discriminator": disc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="latest", help="Run dir or 'latest'")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    run_dir = resolve_run(args.run)
    evaluate(run_dir, args.episodes)
