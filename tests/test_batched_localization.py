import pytest
import torch

from minutia.localization import localize_candidates, localize_candidates_batched
from minutia.localization.gaussian import gaussian_mean


def _movie(
    dtype: torch.dtype = torch.float64,
    photons: float = 1200.0,
    sigma_x: float = 1.15,
    sigma_y: float = 1.65,
    poisson: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.arange(32, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    truth = torch.tensor(
        [16.35, 15.7, photons, sigma_x, sigma_y, 4.0], dtype=dtype
    )
    frame = gaussian_mean(xx, yy, *truth)
    if poisson:
        torch.manual_seed(23)
        frame = torch.poisson(frame)
    return frame.unsqueeze(0), truth


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 2e-3), (torch.float32, 0.15)])
def test_batched_matches_reference_for_subpixel_astigmatic_fit(
    dtype: torch.dtype, atol: float
) -> None:
    frames, truth = _movie(dtype)
    candidates = torch.tensor([[0, 16, 16, 0.9]], dtype=dtype)
    reference = localize_candidates(frames, candidates, iterations=40)
    batched = localize_candidates_batched(frames, candidates, iterations=40)
    assert torch.equal(reference.valid, batched.valid)
    assert torch.allclose(reference.parameters, batched.parameters, atol=atol, rtol=atol)
    assert torch.allclose(reference.log_likelihood, batched.log_likelihood, atol=atol, rtol=atol)
    assert torch.all(torch.isfinite(batched.fisher))
    assert torch.all(torch.isfinite(batched.covariance))
    assert torch.allclose(batched.parameters[0], truth, atol=atol * 5, rtol=atol)


@pytest.mark.parametrize("photons", [80.0, 500.0, 2500.0])
@pytest.mark.parametrize("poisson", [False, True])
def test_batched_matches_reference_across_signal_regimes(
    photons: float, poisson: bool
) -> None:
    frames, _ = _movie(
        photons=photons,
        sigma_x=1.3,
        sigma_y=1.3,
        poisson=poisson,
    )
    candidates = torch.tensor([[0, 16, 16, 0.9]], dtype=torch.float64)
    reference = localize_candidates(frames, candidates, iterations=35)
    batched = localize_candidates_batched(frames, candidates, iterations=35)
    assert torch.equal(reference.valid, batched.valid)
    if bool(reference.valid[0]):
        # Very low-count fits are ill-conditioned; both paths still agree on
        # position and remain within a deliberately wider physical tolerance.
        tolerance = torch.tensor([0.08, 0.08, 150.0, 1.5, 1.5, 0.5])
        assert torch.all(
            (reference.parameters - batched.parameters).abs()
            <= tolerance + 0.1 * reference.parameters.abs()
        )
        assert torch.allclose(
            reference.log_likelihood, batched.log_likelihood, atol=0.2, rtol=0.01
        )


def test_batched_handles_multiple_frames_and_invalid_edges_without_host_flags() -> None:
    frames, _ = _movie()
    frames = torch.cat((frames, frames), dim=0)
    candidates = torch.tensor(
        [[0, 16, 16, 1.0], [1, 16, 16, 0.8], [0, 1, 1, 0.2], [4, 16, 16, 0.1]],
        dtype=torch.float64,
    )
    result = localize_candidates_batched(frames, candidates, iterations=20)
    assert result.valid.device == candidates.device
    assert torch.equal(result.valid, torch.tensor([True, True, False, False]))
    assert torch.isnan(result.parameters[2:]).all()
    assert result.fisher.shape == (4, 6, 6)


def test_empty_batch_preserves_device_and_output_shapes() -> None:
    frames = torch.ones(1, 16, 16)
    candidates = torch.empty((0, 4))
    result = localize_candidates_batched(frames, candidates)
    assert result.parameters.shape == (0, 6)
    assert result.fisher.shape == (0, 6, 6)
