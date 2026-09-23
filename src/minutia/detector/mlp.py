"""The published single-hidden-layer detector."""

import torch
from torch import nn


class CanonicalDetector(nn.Module):
    """7x7 → 30 sigmoid MLP with a scalar sigmoid output."""

    def __init__(self, hidden_units: int = 30) -> None:
        super().__init__()
        self.window_size = 7
        self.hidden_units = hidden_units
        self.hidden = nn.Linear(49, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.shape[-2:] != (7, 7):
            raise ValueError(f"expected (..., 7, 7), got {tuple(patches.shape)}")
        return torch.sigmoid(
            self.output(torch.sigmoid(self.hidden(patches.reshape(*patches.shape[:-2], 49))))
        ).squeeze(-1)

    def score_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Score every valid central pixel without host-side intermediate data."""
        if frames.ndim == 2:
            frames = frames.unsqueeze(0)
        if frames.ndim != 3:
            raise ValueError("frames must have shape (frames, height, width)")
        patches = frames.unfold(1, 7, 1).unfold(2, 7, 1)
        scores = self(patches)
        return torch.nn.functional.pad(scores, (3, 3, 3, 3))
