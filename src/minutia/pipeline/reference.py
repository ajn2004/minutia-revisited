"""Reference detector → coordinate → localizer learning iteration."""

from dataclasses import dataclass

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
) -> IterationResult:
    """Run one complete, readable fit-guided reference iteration.

    Candidate localization and fit validation deliberately use Python scalar
    conversions and per-candidate loops.  That makes this path easy to audit,
    but it is not device-resident end-to-end despite retaining tensors on their
    input device.  A future fast path must provide the same contract without
    these host synchronization points.
    """
    detector_frames = (
        subtract_background(frames, radius=preprocessing_radius, method=preprocessing_method)
        if preprocess
        else frames
    )
    scores = detector.score_frames(detector_frames)
    candidates = select_candidates(scores, threshold=threshold)
    fits = localize_candidates(frames, candidates.as_tensor(), iterations=localization_iterations)
    labels = quality_oracle(fits, candidates.as_tensor(), quality)
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
                float(candidates.score[i]),
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
    return IterationResult(candidates, fits, labels, loss_value)
