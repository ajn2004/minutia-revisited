"""Configurable multi-iteration MINuTIA training loop."""

from dataclasses import dataclass, field

import torch

from minutia.detector import CanonicalDetector
from minutia.quality import QualityConfig
from minutia.training import ReplayBuffer

from .reference import IterationResult, run_iteration


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 10
    minimum_positive_examples: int = 1
    threshold: float = 0.5
    learning_rate: float = 1e-2
    train_steps: int = 5
    max_initialization_attempts: int = 3
    preprocess: bool = True
    preprocessing_method: str = "rolling_ball_approximation"
    preprocessing_radius: int = 5
    quality: QualityConfig = field(default_factory=QualityConfig)


def train_minutia(
    frames: torch.Tensor,
    detector: CanonicalDetector | None = None,
    replay: ReplayBuffer | None = None,
    *,
    config: TrainingConfig | None = None,
) -> tuple[CanonicalDetector, ReplayBuffer, list[IterationResult]]:
    """Run repeated fit-guided updates on a frame tensor.

    This reference uses one supplied movie per iteration. Dataset/frame
    randomization belongs at the experiment layer. Before any training step,
    an explicit finite retry policy prevents an all-negative initialization
    from silently becoming the training set.
    """
    config = config or TrainingConfig()
    if config.minimum_positive_examples < 0:
        raise ValueError("minimum_positive_examples must be non-negative")
    if config.max_initialization_attempts < 1:
        raise ValueError("max_initialization_attempts must be positive")
    detector = detector or CanonicalDetector()
    replay = replay or ReplayBuffer()
    results: list[IterationResult] = []
    for _iteration in range(config.iterations):
        for attempt in range(config.max_initialization_attempts):
            result = run_iteration(
                frames,
                detector,
                replay,
                threshold=config.threshold,
                quality=config.quality,
                learning_rate=config.learning_rate,
                train_steps=config.train_steps,
                preprocess=config.preprocess,
                preprocessing_method=config.preprocessing_method,
                preprocessing_radius=config.preprocessing_radius,
                minimum_positive_examples=config.minimum_positive_examples,
            )
            positives = sum(example.label for example in replay.examples)
            if positives >= config.minimum_positive_examples:
                results.append(result)
                break
            replay.examples.clear()
            if attempt + 1 == config.max_initialization_attempts:
                raise RuntimeError(
                    "minimum positive examples were not produced after "
                    f"{config.max_initialization_attempts} initialization attempts"
                )
            detector.apply(_reset_module)
    return detector, replay, results


def _reset_module(module: torch.nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
