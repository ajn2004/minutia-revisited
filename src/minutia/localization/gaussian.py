"""Pixel-integrated Gaussian PSF and its Poisson mean model."""

import torch


def pixel_integrated_gaussian(
    coordinate: torch.Tensor,
    centre: torch.Tensor | float,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Probability mass of a unit Gaussian in finite pixel ``coordinate``."""
    coordinate = torch.as_tensor(coordinate)
    centre = torch.as_tensor(centre, dtype=coordinate.dtype, device=coordinate.device)
    sigma = torch.as_tensor(sigma, dtype=coordinate.dtype, device=coordinate.device).clamp_min(1e-6)
    scale = (
        torch.sqrt(torch.as_tensor(2.0, dtype=coordinate.dtype, device=coordinate.device)) * sigma
    )
    return 0.5 * (
        torch.erf((coordinate - centre + 0.5) / scale)
        - torch.erf((coordinate - centre - 0.5) / scale)
    )


def gaussian_mean(
    x: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor | float,
    y0: torch.Tensor | float,
    photons: torch.Tensor | float,
    sigma_x: torch.Tensor | float,
    sigma_y: torch.Tensor | float,
    background: torch.Tensor | float,
) -> torch.Tensor:
    """Expected Poisson counts for a finite-pixel separable Gaussian."""
    ex = pixel_integrated_gaussian(x, x0, sigma_x)
    ey = pixel_integrated_gaussian(y, y0, sigma_y)
    return torch.as_tensor(photons, dtype=x.dtype, device=x.device) * ex * ey + background
