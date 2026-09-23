"""Deterministic synthetic microscopy data for validation and examples."""

from dataclasses import dataclass

import torch

from minutia.localization.gaussian import gaussian_mean


@dataclass(frozen=True)
class Molecule:
    frame: int
    x: float
    y: float
    photons: float
    sigma_x: float
    sigma_y: float
    background: float


@dataclass
class SyntheticMovie:
    frames: torch.Tensor
    molecules: list[Molecule]


def simulate_movie(
    *,
    n_frames: int = 4,
    height: int = 32,
    width: int = 32,
    molecules_per_frame: int = 2,
    photons: float = 1200.0,
    background: float = 4.0,
    sigma_x: float = 1.25,
    sigma_y: float | None = None,
    poisson: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> SyntheticMovie:
    """Create frames with known sub-pixel, optionally astigmatic molecules.

    Coordinates use image convention ``x=column`` and ``y=row``. A local
    generator makes seeded output independent of global torch RNG state.
    """
    if sigma_y is None:
        sigma_y = sigma_x
    if min(height, width) < 9:
        raise ValueError("frames must be at least 9 pixels wide for a 7x7 detector")
    generator = torch.Generator().manual_seed(seed)
    frames = torch.full((n_frames, height, width), background, dtype=dtype)
    truth: list[Molecule] = []
    for frame in range(n_frames):
        for _ in range(molecules_per_frame):
            x = float(torch.rand((), generator=generator) * (width - 8) + 4)
            y = float(torch.rand((), generator=generator) * (height - 8) + 4)
            molecule = Molecule(frame, x, y, photons, sigma_x, sigma_y, background)
            truth.append(molecule)
            yy, xx = torch.meshgrid(
                torch.arange(height, dtype=dtype),
                torch.arange(width, dtype=dtype),
                indexing="ij",
            )
            # ``gaussian_mean`` includes its uniform background. Add only the
            # molecule contribution because the frame already has one shared
            # acquisition background, independent of molecule count.
            frames[frame] += gaussian_mean(
                xx, yy, x, y, photons, sigma_x, sigma_y, background
            ) - background
    if poisson:
        frames = torch.poisson(frames, generator=generator)
    return SyntheticMovie(frames, truth)
