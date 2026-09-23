"""Tensor-native accelerator-resident MINuTIA iteration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn

from minutia.detector import CandidateSet, CanonicalDetector, select_candidates
from minutia.localization import FitResult, localize_candidates_batched
from minutia.preprocess import subtract_background
from minutia.quality import QualityConfig, quality_oracle, quality_rejection_counts
from minutia.training import TensorReplayBuffer


@dataclass
class BatchedIterationResult:
    candidates: CandidateSet
    fits: FitResult
    labels: torch.Tensor
    loss: torch.Tensor
    positive_count: torch.Tensor
    candidate_count: torch.Tensor
    timings: dict[str, float]
    quality_rejections: dict[str, int]


def gather_training_patches(
    frames: torch.Tensor, candidates: torch.Tensor, *, radius: int = 3
) -> torch.Tensor:
    """Gather all detector windows with tensor indexing on ``frames.device``."""
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    if frames.ndim != 3 or candidates.ndim != 2 or candidates.shape[1] < 3:
        raise ValueError("frames must be [F,H,W] and candidates must be [N,3+]")
    height, width = frames.shape[-2:]
    frame = candidates[:, 0].to(torch.long).clamp(0, frames.shape[0] - 1)
    x = candidates[:, 1].to(torch.long).clamp(radius, width - radius - 1)
    y = candidates[:, 2].to(torch.long).clamp(radius, height - radius - 1)
    offsets = torch.arange(-radius, radius + 1, device=frames.device, dtype=torch.long)
    yy = y[:, None, None] + offsets[None, :, None]
    xx = x[:, None, None] + offsets[None, None, :]
    return frames[frame[:, None, None], yy, xx]


def run_iteration_batched(
    frames: torch.Tensor,
    detector: CanonicalDetector,
    replay: TensorReplayBuffer,
    *,
    threshold: float = 0.5,
    quality: QualityConfig | None = None,
    learning_rate: float = 1e-2,
    train_steps: int = 1,
    preprocess: bool = True,
    preprocessing_method: str = "rolling_ball_approximation",
    preprocessing_radius: int = 5,
    minimum_positive_examples: int = 0,
    localization_chunk_size: int | None = None,
    localization_iterations: int = 20,
    progress: Callable[[int], None] | None = None,
) -> BatchedIterationResult:
    """Run one MINuTIA iteration without per-candidate host synchronization.

    ``loss`` and the counters are tensors.  Applications may convert them to
    Python values at an explicit reporting boundary after this function returns.
    """
    if frames.device != replay.device:
        raise ValueError("frames and replay must use the same device")
    started = perf_counter()
    preprocess_started = perf_counter()
    detector_frames = (
        subtract_background(
            frames, radius=preprocessing_radius, method=preprocessing_method
        )
        if preprocess
        else frames
    )
    scores = detector.score_frames(detector_frames)
    candidates = select_candidates(scores, threshold=threshold)
    preprocessing_detection_nms_seconds = perf_counter() - preprocess_started
    if progress is not None:
        progress(len(candidates.score))
    candidate_tensor = candidates.as_tensor()
    localization_started = perf_counter()
    fits = localize_candidates_batched(
        frames,
        candidate_tensor,
        iterations=localization_iterations,
        chunk_size=localization_chunk_size,
    )
    localization_seconds = perf_counter() - localization_started
    quality_started = perf_counter()
    labels = quality_oracle(fits, candidate_tensor, quality)
    quality_rejections = quality_rejection_counts(fits, candidate_tensor, quality)
    quality_oracle_seconds = perf_counter() - quality_started
    replay_started = perf_counter()
    patches = gather_training_patches(detector_frames, candidate_tensor)
    replay.add(
        patches,
        labels,
        candidates.frame,
        candidates.x,
        candidates.y,
        candidates.score,
        fits.parameters,
    )

    positive_count = replay.labels.sum(dtype=torch.long)
    candidate_count = torch.as_tensor(candidates.score.shape[0], device=frames.device)
    loss = torch.zeros((), dtype=frames.dtype, device=frames.device)
    if replay.num_examples > 0 and train_steps > 0:
        optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
        detector.train()
        patches_train, target = replay.training_tensors()
        should_train = (positive_count >= minimum_positive_examples).to(frames.dtype)
        for _ in range(train_steps):
            optimizer.zero_grad()
            prediction = detector(patches_train)
            loss = nn.functional.binary_cross_entropy(prediction, target) * should_train
            loss.backward()
            optimizer.step()
    replay_training_seconds = perf_counter() - replay_started
    return BatchedIterationResult(
        candidates=candidates,
        fits=fits,
        labels=labels,
        loss=loss,
        positive_count=positive_count,
        candidate_count=candidate_count,
        timings={
            "preprocessing_detection_nms_seconds": preprocessing_detection_nms_seconds,
            "localization_seconds": localization_seconds,
            "quality_oracle_seconds": quality_oracle_seconds,
            "replay_training_seconds": replay_training_seconds,
            "total_iteration_seconds": perf_counter() - started,
        },
        quality_rejections=quality_rejections,
    )
