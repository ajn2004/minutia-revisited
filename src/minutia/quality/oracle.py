"""Configurable, inspectable fit-quality policies.

The historical policy intentionally mirrors the strict comparisons in
``app_gpu_tol_all_color_learning.m``.  In particular, the MATLAB ``*_crlb``
values are variances, whereas :attr:`FitResult.uncertainty` contains standard
deviations.
"""

from dataclasses import dataclass
from typing import Literal

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
    quality_mode: Literal["modern", "historical"] = "modern"


@dataclass(frozen=True)
class HistoricalQualityConfig:
    """The tolerance values used by the 2018 MATLAB reproduction."""

    quality_mode: Literal["historical"] = "historical"
    photons_min: float = 10.0
    photons_max: float = 5000.0
    background_min: float = -10.0
    background_max: float = 12.0
    sigma_min: float = 1.0
    sigma_max: float = 10.0
    photon_variance_min: float = 0.0
    photon_variance_max: float = 15000.0
    position_variance_min: float = 0.0
    position_variance_max: float = 0.91
    background_variance_min: float = 0.0
    background_variance_max: float = 2.0
    sigma_variance_min: float = 0.0
    sigma_variance_max: float = 2.0
    max_fractional_uncertainty: float = 0.5


QualityPolicy = QualityConfig | HistoricalQualityConfig


def _matlab_round(value: torch.Tensor) -> torch.Tensor:
    """Round halves away from zero, as MATLAB's ``round`` does."""
    return torch.sign(value) * torch.floor(value.abs() + 0.5)


def _historical_masks(
    fits: FitResult, candidates: torch.Tensor, config: HistoricalQualityConfig
) -> dict[str, torch.Tensor]:
    p = fits.parameters
    covariance = fits.covariance
    finite = torch.isfinite(p).all(1) & fits.valid
    if covariance is None:
        variance = torch.full_like(p, torch.nan)
        finite = finite & torch.zeros_like(finite)
    else:
        assert fits.covariance is not None
        variance = fits.covariance.diagonal(dim1=-2, dim2=-1)
        finite = finite & torch.isfinite(variance).all(1)
    # The legacy code computes these from CRLB variances, not from the
    # ``uncertainty`` (standard-deviation) field exposed by FitResult.
    # ``sqrt`` intentionally follows the historical expression; invalid or
    # negative variances are rejected by the variance gates below.
    fractional_n = variance[:, 2].sqrt() / p[:, 2]
    fractional_b = variance[:, 5].sqrt() / p[:, 5]
    fractional_sx = variance[:, 3].sqrt() / p[:, 3]
    fractional_sy = variance[:, 4].sqrt() / p[:, 4]
    return {
        "valid_numerical_fit": finite,
        "photons": (p[:, 2] > config.photons_min) & (p[:, 2] < config.photons_max),
        "background": (p[:, 5] > config.background_min) & (p[:, 5] < config.background_max),
        "sigma": (
            (p[:, 3] > config.sigma_min) & (p[:, 3] < config.sigma_max)
            & (p[:, 4] > config.sigma_min) & (p[:, 4] < config.sigma_max)
        ),
        "photon_crlb_variance": (
            (variance[:, 2] > config.photon_variance_min)
            & (variance[:, 2] < config.photon_variance_max)
        ),
        "position_crlb_variance": (
            (variance[:, 0] > config.position_variance_min)
            & (variance[:, 0] < config.position_variance_max)
            & (variance[:, 1] > config.position_variance_min)
            & (variance[:, 1] < config.position_variance_max)
        ),
        "background_crlb_variance": (
            (variance[:, 5] > config.background_variance_min)
            & (variance[:, 5] < config.background_variance_max)
        ),
        "sigma_crlb_variance": (
            (variance[:, 3] > config.sigma_variance_min)
            & (variance[:, 3] < config.sigma_variance_max)
            & (variance[:, 4] > config.sigma_variance_min)
            & (variance[:, 4] < config.sigma_variance_max)
        ),
        "fractional_uncertainty": (
            (fractional_n < config.max_fractional_uncertainty)
            & (fractional_b < config.max_fractional_uncertainty)
            & (fractional_sx < config.max_fractional_uncertainty)
            & (fractional_sy < config.max_fractional_uncertainty)
        ),
        "position_consistency": (
            (_matlab_round(p[:, 0]) >= candidates[:, 1] - 1)
            & (_matlab_round(p[:, 0]) <= candidates[:, 1] + 1)
            & (_matlab_round(p[:, 1]) >= candidates[:, 2] - 1)
            & (_matlab_round(p[:, 1]) <= candidates[:, 2] + 1)
        ),
    }


def quality_rejection_counts(
    fits: FitResult, candidates: torch.Tensor, config: QualityPolicy | None = None
) -> dict[str, int]:
    """Count candidates rejected by each named criterion.

    Counts are marginal (each criterion is evaluated against every candidate),
    making them useful for diagnosing which tolerance is most restrictive.
    """
    config = config or QualityConfig()
    if isinstance(config, HistoricalQualityConfig) or config.quality_mode == "historical":
        policy = (
            config
            if isinstance(config, HistoricalQualityConfig)
            else HistoricalQualityConfig()
        )
        masks = _historical_masks(fits, candidates, policy)
        return {name: int((~mask).sum()) for name, mask in masks.items()}
    p, u = fits.parameters, fits.uncertainty
    finite = torch.isfinite(p).all(1) & torch.isfinite(u).all(1) & fits.valid
    dx = p[:, 0] - candidates[:, 1]
    dy = p[:, 1] - candidates[:, 2]
    masks = {
        "valid_numerical_fit": finite,
        "photons": (p[:, 2] >= config.photons_min) & (p[:, 2] <= config.photons_max),
        "sigma": (p[:, 3] >= config.sigma_min) & (p[:, 3] <= config.sigma_max)
        & (p[:, 4] >= config.sigma_min) & (p[:, 4] <= config.sigma_max),
        "position_consistency": (dx.abs() <= config.max_position_error)
        & (dy.abs() <= config.max_position_error),
        "fractional_photon_uncertainty": (
            u[:, 2] / p[:, 2] <= config.max_fractional_photon_uncertainty
        ),
        "fractional_sigma_uncertainty": (
            u[:, 3] / p[:, 3] <= config.max_fractional_sigma_uncertainty
        )
        & (u[:, 4] / p[:, 4] <= config.max_fractional_sigma_uncertainty),
    }
    return {name: int((~mask).sum()) for name, mask in masks.items()}


def quality_oracle(
    fits: FitResult, candidates: torch.Tensor, config: QualityPolicy | None = None
) -> torch.Tensor:
    """Return boolean labels; invalid numerical fits are always rejected."""
    config = config or QualityConfig()
    p = fits.parameters
    if p.shape[0] != candidates.shape[0]:
        raise ValueError("one candidate is required for each fit")
    if isinstance(config, HistoricalQualityConfig) or config.quality_mode == "historical":
        policy = (
            config
            if isinstance(config, HistoricalQualityConfig)
            else HistoricalQualityConfig()
        )
        masks = _historical_masks(fits, candidates, policy)
        return torch.stack(tuple(masks.values()), dim=0).all(dim=0)
    p, u = fits.parameters, fits.uncertainty
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
