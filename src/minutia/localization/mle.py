"""Readable reference Poisson MLE with direct source-frame window access."""

from dataclasses import dataclass

import torch

from .gaussian import gaussian_mean


@dataclass
class FitResult:
    """Localization outputs, one row per candidate, in ``x,y,N,sigma_x,sigma_y,b`` order."""

    parameters: torch.Tensor
    uncertainty: torch.Tensor
    log_likelihood: torch.Tensor
    valid: torch.Tensor


def _fit_patch(
    patch: torch.Tensor, iterations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    dtype = patch.dtype
    size = patch.shape[-1]
    grid = torch.arange(size, dtype=dtype, device=patch.device) - (size - 1) / 2
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    background = patch.min().clamp_min(1e-3)
    signal = (patch - background).clamp_min(0)
    total = signal.sum().clamp_min(1e-3)
    x0 = (signal * xx).sum() / total
    y0 = (signal * yy).sum() / total
    # Optimise physically valid values in a transformed parameterization.
    raw = torch.tensor(
        [
            x0,
            y0,
            total.log(),
            torch.tensor(1.2, dtype=dtype, device=patch.device).log(),
            torch.tensor(1.2, dtype=dtype, device=patch.device).log(),
            background.log(),
        ],
        dtype=dtype,
        device=patch.device,
        requires_grad=True,
    )

    def unpack(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (value[0], value[1], value[2].exp(), value[3].exp(), value[4].exp(), value[5].exp())
        )

    optimizer = torch.optim.LBFGS(
        [raw], max_iter=iterations, line_search_fn="strong_wolfe", tolerance_grad=1e-7
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        params = unpack(raw)
        mean = gaussian_mean(
            xx, yy, params[0], params[1], params[2], params[3], params[4], params[5]
        ).clamp_min(1e-8)
        loss = (mean - patch * torch.log(mean)).sum()
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        params = unpack(raw).detach()
        mean = gaussian_mean(xx, yy, *params).clamp_min(1e-8)

        # Fisher information for Poisson observations: J^T diag(1/mu) J.
        def model(p: torch.Tensor) -> torch.Tensor:
            return gaussian_mean(xx, yy, *p).reshape(-1)

        jac = torch.autograd.functional.jacobian(model, params).reshape(-1, 6)
        fisher = jac.T @ (jac / mean.reshape(-1, 1))
        covariance = torch.linalg.pinv(fisher, hermitian=True)
        uncertainty = covariance.diagonal().clamp_min(0).sqrt()
        ll = (patch * torch.log(mean) - mean - torch.lgamma(patch + 1)).sum()
        valid = bool(
            torch.isfinite(params).all()
            and torch.isfinite(uncertainty).all()
            and (params[2:] > 0).all()
        )
        return params, uncertainty, ll, valid
    except (RuntimeError, ValueError):
        nan = torch.full((6,), torch.nan, dtype=dtype, device=patch.device)
        return nan, nan, torch.tensor(float("nan"), dtype=dtype, device=patch.device), False


def localize_candidates(
    frames: torch.Tensor,
    candidates: torch.Tensor,
    *,
    radius: int = 4,
    iterations: int = 20,
) -> FitResult:
    """Fit candidates ``(frame, x, y, score)`` by reading ``frames`` directly.

    This intentionally readable reference loops over candidates. The tensors
    and all optimizer work remain on the input device; batching/vectorization
    is a later optimization with numerical-equivalence tests.
    """
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    if candidates.ndim != 2 or candidates.shape[1] < 3:
        raise ValueError("candidates must have columns frame, x, y[, score]")
    params, errors, likelihoods, validity = [], [], [], []
    height, width = frames.shape[-2:]
    for candidate in candidates:
        frame, x, y = (int(candidate[0]), int(candidate[1]), int(candidate[2]))
        if (
            frame < 0
            or frame >= frames.shape[0]
            or x < radius
            or y < radius
            or x + radius >= width
            or y + radius >= height
        ):
            nan = torch.full((6,), torch.nan, dtype=frames.dtype, device=frames.device)
            params.append(nan)
            errors.append(nan)
            likelihoods.append(nan[0])
            validity.append(False)
            continue
        patch = frames[frame, y - radius : y + radius + 1, x - radius : x + radius + 1]
        fit, error, ll, valid = _fit_patch(patch, iterations)
        # Convert local fitted position to image coordinates.
        fit = fit.clone()
        fit[0] += x
        fit[1] += y
        params.append(fit)
        errors.append(error)
        likelihoods.append(ll)
        validity.append(valid)
    if not params:
        empty = torch.empty((0, 6), dtype=frames.dtype, device=frames.device)
        return FitResult(
            empty,
            empty,
            torch.empty(0, dtype=frames.dtype, device=frames.device),
            torch.empty(0, dtype=torch.bool, device=frames.device),
        )
    return FitResult(
        torch.stack(params),
        torch.stack(errors),
        torch.stack(likelihoods),
        torch.tensor(validity, device=frames.device),
    )
