"""Configurable, inspectable fit-quality policy."""

from dataclasses import dataclass

import torch

from minutia.localization.mle import FitResult


@dataclass(frozen=True)
class QualityConfig:
    photons_min: float = 10.0
    photons_max: float = 5000.0
    sigma_min: float = 0.5
    sigma_max: float = 4.0
    max_position_error: float = 1.0
    max_fractional_photon_uncertainty: float = 0.5
    max_fractional_sigma_uncertainty: float = 0.5


def quality_oracle(
    fits: FitResult, candidates: torch.Tensor, config: QualityConfig | None = None
) -> torch.Tensor:
    """Return boolean labels; invalid numerical fits are always rejected."""
    config = config or QualityConfig()
    p, u = fits.parameters, fits.uncertainty
    if p.shape[0] != candidates.shape[0]:
        raise ValueError("one candidate is required for each fit")
    dx = p[:, 0] - candidates[:, 1]
    dy = p[:, 1] - candidates[:, 2]
    finite = torch.isfinite(p).all(1) & torch.isfinite(u).all(1) & fits.valid
    return (
        finite
        & (p[:, 2] >= config.photons_min)
        & (p[:, 2] <= config.photons_max)
        & (p[:, 3] >= config.sigma_min)
        & (p[:, 3] <= config.sigma_max)
        & (p[:, 4] >= config.sigma_min)
        & (p[:, 4] <= config.sigma_max)
        & (dx.abs() <= config.max_position_error)
        & (dy.abs() <= config.max_position_error)
        & (u[:, 2] / p[:, 2] <= config.max_fractional_photon_uncertainty)
        & (u[:, 3] / p[:, 3] <= config.max_fractional_sigma_uncertainty)
        & (u[:, 4] / p[:, 4] <= config.max_fractional_sigma_uncertainty)
    )
