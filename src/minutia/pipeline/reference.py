"""Reference detector → coordinate → localizer learning iteration."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from minutia.detector import CandidateSet, CanonicalDetector, select_candidates
from minutia.localization import FitResult, localize_candidates
from minutia.preprocess import subtract_background
from minutia.quality import QualityConfig, quality_oracle
from minutia.training import ReplayBuffer, TrainingExample


@dataclass
class IterationResult:
    candidates: CandidateSet
    fits: FitResult
    labels: torch.Tensor
    loss: float | None
    timings: dict[str, float]


def run_iteration(
    frames: torch.Tensor,
    detector: CanonicalDetector,
    replay: ReplayBuffer,
    *,
    threshold: float = 0.5,
    quality: QualityConfig | None = None,
    learning_rate: float = 1e-2,
    train_steps: int = 1,
    preprocess: bool = True,
    preprocessing_method: str = "rolling_ball_approximation",
    preprocessing_radius: int = 5,
    minimum_positive_examples: int = 0,
    localization_iterations: int = 20,
    progress: Callable[[int], None] | None = None,
) -> IterationResult:
    """Run one complete, readable fit-guided reference iteration.

    Candidate localization and fit validation deliberately use Python scalar
    conversions and per-candidate loops.  That makes this path easy to audit,
    but it is not device-resident end-to-end despite retaining tensors on their
    input device.  A future fast path must provide the same contract without
    these host synchronization points.
    """
    started = perf_counter()
    preprocessing_started = perf_counter()
    detector_frames = (
        subtract_background(frames, radius=preprocessing_radius, method=preprocessing_method)
        if preprocess
        else frames
    )
    scores = detector.score_frames(detector_frames)
    candidates = select_candidates(scores, threshold=threshold)
    preprocessing_detection_nms_seconds = perf_counter() - preprocessing_started
    if progress is not None:
        progress(len(candidates.score))
    localization_started = perf_counter()
    fits = localize_candidates(frames, candidates.as_tensor(), iterations=localization_iterations)
    localization_seconds = perf_counter() - localization_started
    quality_started = perf_counter()
    labels = quality_oracle(fits, candidates.as_tensor(), quality)
    quality_oracle_seconds = perf_counter() - quality_started
    replay_started = perf_counter()
    examples: list[TrainingExample] = []
    for i in range(len(candidates.score)):
        f, x, y = (int(candidates.frame[i]), int(candidates.x[i]), int(candidates.y[i]))
        examples.append(
            TrainingExample(
                detector_frames[f, y - 3 : y + 4, x - 3 : x + 4].detach(),
                bool(labels[i]),
                f,
                x,
                y,
                float(candidates.score[i].detach()),
                fits.parameters[i].detach(),
            )
        )
    replay.add(examples)
    loss_value: float | None = None
    positives = sum(example.label for example in replay.examples)
    if replay.examples and train_steps > 0 and positives >= minimum_positive_examples:
        optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
        detector.train()
        patches, target = replay.tensors(device=frames.device)
        for _ in range(train_steps):
            optimizer.zero_grad()
            prediction = detector(patches)
            loss = nn.functional.binary_cross_entropy(prediction, target)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
    return IterationResult(
        candidates,
        fits,
        labels,
        loss_value,
        {
            "preprocessing_detection_nms_seconds": preprocessing_detection_nms_seconds,
            "localization_seconds": localization_seconds,
            "quality_oracle_seconds": quality_oracle_seconds,
            "replay_training_seconds": perf_counter() - replay_started,
            "total_iteration_seconds": perf_counter() - started,
        },
    )
