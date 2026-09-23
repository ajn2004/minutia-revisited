"""Explicit background preprocessing for the readable reference path."""

from torch import Tensor
from torch.nn import functional as F

SUPPORTED_METHODS = ("rolling_ball_approximation", "local_mean")


def subtract_background(
    frames: Tensor, radius: int = 5, *, method: str = "rolling_ball_approximation"
) -> Tensor:
    """Subtract a smooth local background, preserving device and dtype.

    ``rolling_ball_approximation`` is the default canonical-reference target,
    but is not a literal rolling-ball implementation: it uses a local mean as
    a portable approximation. ``local_mean`` is an explicit experimental alias.
    A calibrated rolling-ball implementation can replace the approximation
    without changing the detector/localizer contract.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
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
