"""Device-compatible thresholding and local-maximum candidate compaction."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CandidateSet:
    frame: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    score: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack((self.frame, self.x, self.y, self.score), dim=1)


def select_candidates(
    scores: torch.Tensor, threshold: float = 0.5, neighbourhood: int = 5
) -> CandidateSet:
    """Return thresholded local maxima; all outputs stay on ``scores.device``."""
    if scores.ndim == 2:
        scores = scores.unsqueeze(0)
    if scores.ndim != 3 or neighbourhood % 2 == 0:
        raise ValueError("scores must be (frames, height, width), with odd neighbourhood")
    pooled = F.max_pool2d(
        scores.unsqueeze(1), neighbourhood, stride=1, padding=neighbourhood // 2
    ).squeeze(1)
    keep = (scores >= threshold) & (scores == pooled)
    frame, y, x = torch.where(keep)
    return CandidateSet(frame, x, y, scores[frame, y, x])
