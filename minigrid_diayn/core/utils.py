"""Utility functions for training and experiment management."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def get_device() -> str:
    """Return best available compute device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    """Set random seeds across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


RUNS_DIR = Path(__file__).parent.parent / "runs"


def get_run_dir(env_key: str, base_dir: Path = None) -> Path:
    """Create timestamped run directory."""
    base = base_dir or RUNS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{env_key}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_latest_run(env_filter: str = None, base_dir: Path = None) -> Path:
    """Find most recent run directory, optionally filtered by env."""
    base = base_dir or RUNS_DIR
    if not base.exists():
        raise FileNotFoundError(f"No runs directory: {base}")

    runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

    for run in runs:
        if not run.is_dir():
            continue
        if env_filter and not run.name.startswith(env_filter):
            continue
        if not (run / "config.json").exists():
            continue
        return run

    raise FileNotFoundError(
        "No runs found"
        + (f" for env '{env_filter}'" if env_filter else "")
        + " with config.json"
    )


def resolve_run(run_arg: str, base_dir: Path = None) -> Path:
    """Resolve run argument to path. Accepts 'latest', env name, or full path."""
    if run_arg == "latest":
        return get_latest_run(base_dir=base_dir)

    path = Path(run_arg)
    if path.exists():
        return path

    # Try as env filter
    return get_latest_run(env_filter=run_arg, base_dir=base_dir)
