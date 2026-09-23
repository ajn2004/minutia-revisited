"""Reference detector → coordinate → localizer learning iteration."""

from dataclasses import dataclass

import torch
from torch import nn

from minutia.detector import CandidateSet, CanonicalDetector, select_candidates
from minutia.localization import FitResult, localize_candidates
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
) -> IterationResult:
    """Run one complete fit-guided iteration without host-visible intermediates."""
    scores = detector.score_frames(frames)
    candidates = select_candidates(scores, threshold=threshold)
    fits = localize_candidates(frames, candidates.as_tensor())
    labels = quality_oracle(fits, candidates.as_tensor(), quality)
    examples: list[TrainingExample] = []
    for i in range(len(candidates.score)):
        f, x, y = (int(candidates.frame[i]), int(candidates.x[i]), int(candidates.y[i]))
        examples.append(
            TrainingExample(
                frames[f, y - 3 : y + 4, x - 3 : x + 4].detach(),
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
    if replay.examples and train_steps > 0:
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
