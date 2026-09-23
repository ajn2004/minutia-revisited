"""TOML configuration for the Figure 5 reproduction."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    provenance: str
    seed: int
    n_frames: int
    height: int
    width: int
    molecules_per_frame: int
    iterations: int
    frames_per_iteration: int
    photon_mean: float
    photon_std: float
    background_mean: float
    background_std: float
    sigma_x: float
    sigma_y: float
    sigma_std: float
    poisson: bool
    matching_radius: float
    detector_threshold: float
    mle_iterations: int
    train_steps: int
    learning_rate: float
    toss_positive_fraction: float
    replay_capacity: int
    schedule: str = "paper_methods"
    execution_path: str = "batched"
    initialization: str = "historical_uniform"
    initialization_epsilon: float = 0.12
    bootstrap_enabled: bool = True
    bootstrap_min_positives: int = 40
    bootstrap_max_attempts: int = 100
    training_mode: str = "modern_adam"

    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("rb") as handle:
            values = tomllib.load(handle)
        return cls(**values)


def frames_for_iteration(schedule: str, iteration: int, available: int) -> int:
    """Return the historical frame count (iteration numbers are one-based)."""
    if schedule == "paper_methods":
        requested = 10 if iteration < 15 else 1000
    elif schedule == "legacy_matlab":
        if iteration < 5:
            requested = 10
        elif iteration < 15:
            requested = 50
        elif iteration < 25:
            requested = 500
        else:
            requested = 1000
    else:
        raise ValueError(f"unknown training schedule: {schedule}")
    return min(requested, available)
