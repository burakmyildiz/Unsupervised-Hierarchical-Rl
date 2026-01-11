"""DIAYN training script matching reference implementation."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from core import DIAYNConfig, make_env, get_run_dir, set_seed
from agents import DIAYNAgent


def train(config: DIAYNConfig, run_dir: Path):
    set_seed(config.seed)

    movement_only = getattr(config, 'movement_only', False)
    env, env_info = make_env(
        config.env_key,
        seed=config.seed,
        partial_obs=config.partial_obs,
        random_start=config.random_start,
        movement_only=movement_only
    )

    print("=" * 60)
    print(f"Environment: {env_info['env_name']}")
    print(f"Grid: {env_info['grid_size']}x{env_info['grid_size']}")
    print(f"Obs dim: {env_info['obs_dim']}, Actions: {env_info['num_actions']}")
    print(f"Partial obs: {config.partial_obs}")
    print(f"Movement only: {movement_only}")
    print(f"Random start: {config.random_start}")
    print(f"Skills: {config.num_skills}, Device: {config.device}")
    print(f"Episodes: {config.num_episodes}, Steps/ep: {config.max_steps}")
    print(f"Entropy coef: {config.entropy_coef}")
    print(f"Feature dim: {config.feature_dim}, Hidden dim: {config.hidden_dim}")
    print(f"Run dir: {run_dir}")
    print("=" * 60)

    agent = DIAYNAgent(
        config, env_info["obs_dim"], env_info["num_actions"], env_info["grid_size"],
        partial_obs=config.partial_obs
    )

    # Setup directories
    diayn_dir = run_dir / "diayn"
    diayn_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config.save(run_dir / "config.json")

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "discriminator_loss": [],
        "discriminator_accuracy": [],
        "policy_loss": [],
        "entropy": [],
        "collection_accuracy": [],
    }

    total_steps = 0
    best_acc = 0.0

    pbar = tqdm(range(config.num_episodes), desc="Training")
    for ep in pbar:
        skill = np.random.randint(config.num_skills)
        state, info = env.reset()

        ep_reward = 0.0
        ep_correct = 0

        # Collect episode
        for step in range(config.max_steps):
            if total_steps < config.start_training_step:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, skill)

            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get position for logging (not used by discriminator anymore)
            position = info["position_normalized"]

            # Compute reward for logging
            diag = agent.compute_pseudo_reward_with_diagnostics(next_state, position, skill)
            ep_correct += int(diag["is_correct"])
            ep_reward += diag["reward"]

            # Add to replay buffer
            agent.replay_buffer.push(state, action, diag["reward"], next_state, done, skill, position)

            total_steps += 1
            state = next_state
            if done:
                break

        ep_len = step + 1
        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_lengths"].append(ep_len)
        metrics["collection_accuracy"].append(ep_correct / ep_len)

        # Training phase - single update per episode (like reference)
        if total_steps >= config.start_training_step:
            # Optionally freeze discriminator for first N episodes
            freeze_disc = getattr(config, 'freeze_disc_episodes', 0)
            train_disc = (ep >= freeze_disc)

            for _ in range(config.updates_per_episode):
                disc_loss, disc_acc, policy_loss, entropy = agent.update(config.batch_size, train_discriminator=train_disc)

            metrics["discriminator_loss"].append(disc_loss)
            metrics["discriminator_accuracy"].append(disc_acc)
            metrics["policy_loss"].append(policy_loss)
            metrics["entropy"].append(entropy)

            # Step LR schedulers
            agent.step_schedulers()

        if (ep + 1) % config.log_interval == 0 and total_steps >= config.start_training_step:
            train_acc = np.mean(metrics["discriminator_accuracy"][-config.log_interval:])
            coll_acc = np.mean(metrics["collection_accuracy"][-config.log_interval:])
            best_acc = max(best_acc, train_acc)
            pbar.set_postfix_str(
                f"r={np.mean(metrics['episode_rewards'][-config.log_interval:]):.1f} "
                f"tr={train_acc:.2f} co={coll_acc:.2f} H={np.mean(metrics['entropy'][-config.log_interval:]):.2f}"
            )

        if (ep + 1) % config.save_interval == 0:
            agent.save(diayn_dir / f"checkpoint_ep{ep+1}.pt")

    # Save final
    agent.save(diayn_dir / "final_model.pt")

    with open(diayn_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    env.close()

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best training accuracy: {best_acc:.4f}")
    print(f"Final reward (last 100): {np.mean(metrics['episode_rewards'][-100:]):.1f}")
    final_train_acc = np.mean(metrics["discriminator_accuracy"][-100:])
    final_coll_acc = np.mean(metrics["collection_accuracy"][-100:])
    print(f"Final train_acc (last 100): {final_train_acc:.4f}")
    print(f"Final coll_acc (last 100): {final_coll_acc:.4f}")
    if final_train_acc < 0.3:
        print("WARNING: Low accuracy - skills may not be differentiating well")
    print(f"Run saved to: {run_dir}")
    print("=" * 60)

    return agent, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="empty-8x8")
    parser.add_argument("--num_skills", type=int, default=8)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--random_start", action="store_true",
                        help="Enable random starting position")
    parser.add_argument("--partial_obs", action="store_true",
                        help="Use partial 7x7 observation (like reference)")
    parser.add_argument("--disc_type", type=str, default="state", choices=["state", "position"],
                        help="Discriminator type: 'state' (reference) or 'position' (geometric init)")
    parser.add_argument("--freeze_disc", type=int, default=0,
                        help="Freeze discriminator for first N episodes (for position disc bootstrap)")
    parser.add_argument("--movement_only", action="store_true",
                        help="Restrict actions to {left, right, forward} - removes no-ops")
    args = parser.parse_args()

    config = DIAYNConfig(
        env_key=args.env,
        num_skills=args.num_skills,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        entropy_coef=args.entropy_coef,
        random_start=args.random_start,
        partial_obs=args.partial_obs,
        discriminator_type=args.disc_type,
        freeze_disc_episodes=args.freeze_disc,
        movement_only=args.movement_only,
    )

    run_dir = get_run_dir(config.env_key)
    train(config, run_dir)
