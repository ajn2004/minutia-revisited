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
    randomization belongs at the experiment layer, while the positive-example
    safeguard mirrors the historical first-iteration behavior.
    """
    config = config or TrainingConfig()
    detector = detector or CanonicalDetector()
    replay = replay or ReplayBuffer()
    results: list[IterationResult] = []
    for iteration in range(config.iterations):
        result = run_iteration(
            frames,
            detector,
            replay,
            threshold=config.threshold,
            quality=config.quality,
            learning_rate=config.learning_rate,
            train_steps=config.train_steps,
        )
        results.append(result)
        positives = sum(example.label for example in replay.examples)
        if positives >= config.minimum_positive_examples:
            continue
        if iteration == 0:
            # Avoid training a detector forever on an all-negative first buffer.
            detector.apply(_reset_module)
            replay.examples.clear()
    return detector, replay, results


def _reset_module(module: torch.nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
