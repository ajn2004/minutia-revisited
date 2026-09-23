"""Portable tensor-batched Poisson localization.

This module deliberately has no Python loop over candidates.  It is a portable
PyTorch implementation intended to be the numerical stepping stone between
the readable reference localizer and a future fused kernel.
"""

from __future__ import annotations

import math

import torch

from .mle import FitResult


def _pixel_mass_and_derivatives(
    coordinate: torch.Tensor, centre: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pixel mass and derivatives with respect to centre and sigma."""
    root_two = math.sqrt(2.0)
    root_pi = math.sqrt(math.pi)
    sigma = sigma.clamp_min(1e-4)
    upper = (coordinate - centre + 0.5) / (root_two * sigma)
    lower = (coordinate - centre - 0.5) / (root_two * sigma)
    upper_exp = torch.exp(-upper.square())
    lower_exp = torch.exp(-lower.square())
    mass = 0.5 * (torch.erf(upper) - torch.erf(lower))
    d_centre = (lower_exp - upper_exp) / (root_two * root_pi * sigma)
    d_sigma = (lower * lower_exp - upper * upper_exp) / (root_pi * sigma)
    return mass, d_centre, d_sigma


def _model_and_jacobian(
    grid_x: torch.Tensor, grid_y: torch.Tensor, parameters: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the mean and its physical-parameter Jacobian."""
    x0, y0, photons, sigma_x, sigma_y, background = parameters.unbind(-1)
    ex, dex, dsx = _pixel_mass_and_derivatives(grid_x, x0[:, None, None], sigma_x[:, None, None])
    ey, dey, dsy = _pixel_mass_and_derivatives(grid_y, y0[:, None, None], sigma_y[:, None, None])
    mean = photons[:, None, None] * ex * ey + background[:, None, None]
    jacobian = torch.stack(
        (
            photons[:, None, None] * dex * ey,
            photons[:, None, None] * ex * dey,
            ex * ey,
            photons[:, None, None] * dsx * ey,
            photons[:, None, None] * ex * dsy,
            torch.ones_like(mean),
        ),
        dim=-1,
    )
    return mean, jacobian


def _gather_windows(
    frames: torch.Tensor, candidates: torch.Tensor, radius: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather [N,K,K] windows and return the tensor validity mask."""
    frame_count, height, width = frames.shape
    x = candidates[:, 1].to(torch.long)
    y = candidates[:, 2].to(torch.long)
    frame = candidates[:, 0].to(torch.long)
    offsets = torch.arange(-radius, radius + 1, device=frames.device, dtype=torch.long)
    yy = (y[:, None, None] + offsets[None, :, None]).clamp(0, height - 1)
    xx = (x[:, None, None] + offsets[None, None, :]).clamp(0, width - 1)
    frame = frame.clamp(0, frame_count - 1)
    windows = frames[frame[:, None, None], yy, xx]
    valid = (
        (candidates[:, 0] >= 0)
        & (candidates[:, 0] < frame_count)
        & (candidates[:, 1] >= radius)
        & (candidates[:, 2] >= radius)
        & (candidates[:, 1] + radius < width)
        & (candidates[:, 2] + radius < height)
    )
    return windows, valid


def localize_candidates_batched(
    frames: torch.Tensor,
    candidates: torch.Tensor,
    *,
    radius: int = 4,
    iterations: int = 20,
    damping: float = 1e-5,
    chunk_size: int | None = None,
) -> FitResult:
    """Localize all ``(frame, x, y, score)`` candidates with tensor operations.

    Candidate coordinates, intermediate parameters, validity masks, and fit
    outputs remain tensors on ``frames.device``.  Invalid windows are gathered
    from clamped coordinates solely to keep the operation shape-safe; their
    results are masked to NaN in the returned tensors.
    """
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    if frames.ndim != 3 or candidates.ndim != 2 or candidates.shape[1] < 4:
        raise ValueError("frames must be [F,H,W] and candidates must be [N,4]")
    if frames.device != candidates.device:
        raise ValueError("frames and candidates must be on the same device")
    if not frames.is_floating_point():
        raise TypeError("frames must have a floating point dtype")
    if radius < 1 or iterations < 0 or damping <= 0:
        raise ValueError("radius, iterations, and damping must be positive")

    n = candidates.shape[0]
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive when provided")
    if chunk_size is not None and n > chunk_size:
        chunks = [
            localize_candidates_batched(
                frames,
                candidates[start : start + chunk_size],
                radius=radius,
                iterations=iterations,
                damping=damping,
            )
            for start in range(0, n, chunk_size)
        ]
        return FitResult(
            torch.cat(tuple(chunk.parameters for chunk in chunks), dim=0),
            torch.cat(tuple(chunk.uncertainty for chunk in chunks), dim=0),
            torch.cat(tuple(chunk.log_likelihood for chunk in chunks), dim=0),
            torch.cat(tuple(chunk.valid for chunk in chunks), dim=0),
            torch.cat(tuple(chunk.fisher for chunk in chunks), dim=0),
            torch.cat(tuple(chunk.covariance for chunk in chunks), dim=0),
        )
    dtype, device = frames.dtype, frames.device
    if n == 0:
        empty = torch.empty((0, 6), dtype=dtype, device=device)
        return FitResult(empty, empty, torch.empty(0, dtype=dtype, device=device),
                         torch.empty(0, dtype=torch.bool, device=device),
                         torch.empty((0, 6, 6), dtype=dtype, device=device),
                         torch.empty((0, 6, 6), dtype=dtype, device=device))

    patches, valid = _gather_windows(frames, candidates, radius)
    size = 2 * radius + 1
    axis = torch.arange(size, dtype=dtype, device=device) - radius
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    grid_x = grid_x.expand(n, -1, -1)
    grid_y = grid_y.expand(n, -1, -1)
    background = patches.amin(dim=(-2, -1)).clamp_min(1e-3)
    signal = (patches - background[:, None, None]).clamp_min(0)
    total = signal.sum(dim=(-2, -1)).clamp_min(1e-3)
    x0 = (signal * grid_x).sum(dim=(-2, -1)) / total
    y0 = (signal * grid_y).sum(dim=(-2, -1)) / total
    parameters = torch.stack(
        (
            x0,
            y0,
            total,
            torch.full_like(total, 1.2),
            torch.full_like(total, 1.2),
            background,
        ),
        dim=-1,
    )
    finite = torch.isfinite(patches).all(dim=(-2, -1)) & valid
    eye = torch.eye(6, dtype=dtype, device=device).expand(n, -1, -1)
    damping_tensor = torch.as_tensor(damping, dtype=dtype, device=device)

    for _ in range(iterations):
        mean, jacobian = _model_and_jacobian(grid_x, grid_y, parameters)
        safe_mean = mean.clamp_min(1e-6)
        residual = patches / safe_mean - 1.0
        fisher = torch.einsum("nhwk,nhwl,nhw->nkl", jacobian, jacobian, 1.0 / safe_mean)
        gradient = torch.einsum("nhwk,nhw->nk", jacobian, residual)
        step = torch.linalg.pinv(
            fisher + damping_tensor * eye, hermitian=True
        ) @ gradient.unsqueeze(-1)
        step = step.squeeze(-1).nan_to_num(0.0).clamp(-100.0, 100.0)
        proposed = parameters + step
        proposed = torch.cat((proposed[..., :2], proposed[..., 2:].clamp_min(1e-4)), dim=-1)
        parameters = torch.where(finite[:, None], proposed, parameters)
        finite = finite & torch.isfinite(parameters).all(dim=-1)

    mean, jacobian = _model_and_jacobian(grid_x, grid_y, parameters)
    safe_mean = mean.clamp_min(1e-8)
    fisher = torch.einsum("nhwk,nhwl,nhw->nkl", jacobian, jacobian, 1.0 / safe_mean)
    covariance = torch.linalg.pinv(fisher, hermitian=True)
    uncertainty = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    log_likelihood = (
        patches * safe_mean.log() - safe_mean - torch.lgamma(patches + 1)
    ).sum(dim=(-2, -1))
    finite = finite & torch.isfinite(uncertainty).all(dim=-1)
    finite = finite & (parameters[..., 2:] > 0).all(dim=-1)
    parameters = parameters.clone()
    parameters[..., 0] += candidates[:, 1]
    parameters[..., 1] += candidates[:, 2]
    nan_params = torch.full_like(parameters, torch.nan)
    nan_uncertainty = torch.full_like(uncertainty, torch.nan)
    parameters = torch.where(finite[:, None], parameters, nan_params)
    uncertainty = torch.where(finite[:, None], uncertainty, nan_uncertainty)
    log_likelihood = torch.where(finite, log_likelihood, torch.full_like(log_likelihood, torch.nan))
    matrix_nan = torch.full_like(fisher, torch.nan)
    fisher = torch.where(finite[:, None, None], fisher, matrix_nan)
    covariance = torch.where(finite[:, None, None], covariance, matrix_nan)
    return FitResult(parameters, uncertainty, log_likelihood, finite, fisher, covariance)
