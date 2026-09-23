"""Seeded movies with stored truth for the reproduction experiments."""

from dataclasses import dataclass

import torch

from minutia.localization.gaussian import gaussian_mean
from minutia.sim import Molecule

from .config import ExperimentConfig


@dataclass(frozen=True)
class GroundTruthMovie:
    frames: torch.Tensor
    molecules: tuple[Molecule, ...]


def generate_movie(config: ExperimentConfig) -> GroundTruthMovie:
    generator = torch.Generator().manual_seed(config.seed)
    dtype = torch.float64
    yy, xx = torch.meshgrid(
        torch.arange(config.height, dtype=dtype),
        torch.arange(config.width, dtype=dtype),
        indexing="ij",
    )
    frames = torch.zeros((config.n_frames, config.height, config.width), dtype=dtype)
    truths: list[Molecule] = []
    for frame in range(config.n_frames):
        background = max(
            0.1,
            float(
                config.background_mean
                + config.background_std * torch.randn((), generator=generator)
            ),
        )
        for _ in range(config.molecules_per_frame):
            x = float(torch.rand((), generator=generator) * (config.width - 8) + 4)
            y = float(torch.rand((), generator=generator) * (config.height - 8) + 4)
            photons = max(
                5.0,
                float(
                    config.photon_mean + config.photon_std * torch.randn((), generator=generator)
                ),
            )
            sx = max(
                0.55,
                float(config.sigma_x + config.sigma_std * torch.randn((), generator=generator)),
            )
            sy = max(
                0.55,
                float(config.sigma_y + config.sigma_std * torch.randn((), generator=generator)),
            )
            truths.append(Molecule(frame, x, y, photons, sx, sy, background))
            frames[frame] += gaussian_mean(xx, yy, x, y, photons, sx, sy, background) - background
        frames[frame] += background
    if config.poisson:
        frames = torch.poisson(frames, generator=generator)
    return GroundTruthMovie(frames, tuple(truths))
