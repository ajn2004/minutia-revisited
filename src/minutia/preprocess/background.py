"""Background correction primitives."""

import torch
import torch.nn.functional as F


def subtract_background(frames: torch.Tensor, radius: int = 5) -> torch.Tensor:
    """Subtract a smooth local background, preserving device and dtype.

    This portable approximation is intentionally explicit. The publication's
    rolling-ball method can be added as a calibrated backend without changing
    the detector/localizer contract.
    """
    if radius < 1:
        raise ValueError("radius must be positive")
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    if frames.ndim != 3:
        raise ValueError("frames must have shape (frames, height, width)")
    size = 2 * radius + 1
    background = F.avg_pool2d(
        frames.unsqueeze(1), size, stride=1, padding=radius, count_include_pad=False
    )
    return frames - background.squeeze(1)
