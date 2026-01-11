# DIAYN for MiniGrid

Unsupervised skill discovery using Diversity Is All You Need (DIAYN) in discrete grid-world environments.

## Overview

This implementation adapts DIAYN for MiniGrid environments, discovering diverse navigation skills without external rewards. A hierarchical controller can then compose these skills for goal-reaching tasks.

## Project Structure

```
minigrid_diayn/
├── core/                   # Core utilities
│   ├── config.py          # Configuration dataclasses
│   ├── env.py             # Environment wrappers
│   ├── replay_buffer.py   # Experience replay
│   └── utils.py           # Helper functions
├── networks/              # Neural network modules
│   ├── encoders.py        # CNN encoders for grid observations
│   ├── policy.py          # Categorical policy network
│   ├── discriminator.py   # Skill discriminator
│   └── meta.py            # Meta-controller networks
├── agents/                # Agent implementations
│   ├── diayn_agent.py     # DIAYN skill discovery agent
│   └── hierarchical_agent.py  # Meta-controller over skills
├── scripts/               # Training and evaluation
│   ├── train.py           # DIAYN training script
│   ├── evaluate.py        # Skill evaluation
│   ├── visualize.py       # Visualization tools
│   └── train_hierarchical.py  # Hierarchical training
└── tex/                   # LaTeX report and figures
```

## Installation

```bash
pip install gymnasium minigrid torch numpy matplotlib seaborn
```

## Usage

### Train DIAYN Skills

```bash
# Empty 8x8 grid with movement-only actions
python scripts/train.py --env empty-8x8 --num_skills 8 --num_episodes 10000 \
    --max_steps 30 --entropy_coef 0.5 --movement_only --partial_obs

# FourRooms environment
python scripts/train.py --env fourrooms --num_skills 8 --num_episodes 10000 \
    --max_steps 50 --entropy_coef 0.5 --movement_only --partial_obs
```

### Visualize Learned Skills

```bash
python scripts/visualize.py --run latest --all
```

### Train Hierarchical Controller

```bash
python scripts/train_hierarchical.py --run latest --episodes 500
```

## Key Features

- **CNN Encoder**: Processes 7x7x3 partial observations
- **Categorical Policy**: Discrete action distribution with entropy regularization
- **State-based Discriminator**: Classifies skills from encoded visual features
- **Movement-only Mode**: Restricts actions to prevent "camping" equilibrium
- **Hierarchical Control**: Meta-controller composes skills for goal-reaching

## Configuration

Key hyperparameters in `DIAYNConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_skills` | 8 | Number of skills to discover |
| `entropy_coef` | 0.5 | Entropy bonus (higher for discrete actions) |
| `max_steps` | 30 | Episode length |
| `movement_only` | True | Restrict to movement actions |
| `partial_obs` | True | Use 7x7 agent-centric view |

## Results

With proper configuration, DIAYN discovers spatially diverse navigation skills:
- Skills partition the grid into distinct regions
- Movement-only restriction prevents degenerate solutions
- ~55-60% discriminator accuracy indicates meaningful skill differentiation

## References

- [DIAYN Paper](https://arxiv.org/abs/1802.06070): Eysenbach et al., "Diversity is All You Need"
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid): Lightweight grid-world environments
