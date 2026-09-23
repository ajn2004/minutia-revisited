import pytest
import torch

from minutia.detector import CandidateSet, CanonicalDetector, select_candidates
from minutia.localization import FitResult, localize_candidates
from minutia.localization.gaussian import gaussian_mean, pixel_integrated_gaussian
from minutia.pipeline.iterative import TrainingConfig, train_minutia
from minutia.pipeline.reference import IterationResult
from minutia.preprocess import subtract_background
from minutia.sim import simulate_movie


def test_pixel_integrated_gaussian_is_symmetric() -> None:
    coordinates = torch.arange(-8, 9, dtype=torch.float64)
    values = pixel_integrated_gaussian(coordinates, 0.0, 1.2)
    assert torch.allclose(values, values.flip(0))
    assert torch.isclose(values.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)


def test_simulation_is_seeded_and_astigmatic() -> None:
    first = simulate_movie(n_frames=1, molecules_per_frame=1, sigma_x=1.0, sigma_y=1.8, seed=4)
    second = simulate_movie(n_frames=1, molecules_per_frame=1, sigma_x=1.0, sigma_y=1.8, seed=4)
    assert torch.equal(first.frames, second.frames)
    assert first.molecules[0].sigma_x != first.molecules[0].sigma_y


def test_detector_has_published_shapes_and_manual_forward() -> None:
    detector = CanonicalDetector()
    assert tuple(detector.hidden.weight.shape) == (30, 49)
    assert tuple(detector.output.weight.shape) == (1, 30)
    patches = torch.zeros(2, 7, 7)
    expected = torch.sigmoid(
        detector.output(torch.sigmoid(detector.hidden(torch.zeros(2, 49))))
    ).squeeze(1)
    assert torch.allclose(detector(patches), expected)


def test_local_maximum_candidate_selection() -> None:
    scores = torch.zeros(1, 11, 11)
    scores[0, 5, 5] = 0.9
    scores[0, 5, 6] = 0.8
    candidates = select_candidates(scores)
    assert len(candidates.score) == 1
    assert (int(candidates.x[0]), int(candidates.y[0])) == (5, 5)


def test_gaussian_mean_has_expected_background() -> None:
    grid = torch.arange(9, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    mean = gaussian_mean(xx, yy, 4.0, 4.0, 100.0, 1.2, 1.2, 3.0)
    assert torch.all(mean >= 3.0)
    assert mean[4, 4] > mean[0, 0]


def test_mle_recovers_known_noise_free_synthetic_parameters() -> None:
    movie = simulate_movie(
        n_frames=1,
        height=32,
        width=32,
        molecules_per_frame=1,
        photons=1600.0,
        background=4.0,
        sigma_x=1.25,
        sigma_y=1.6,
        poisson=False,
        seed=8,
    )
    truth = movie.molecules[0]
    candidate = torch.tensor(
        [[truth.frame, round(truth.x), round(truth.y), 1.0]], dtype=torch.float64
    )
    fit = localize_candidates(movie.frames, candidate, iterations=50)
    estimate = fit.parameters[0]
    assert bool(fit.valid[0])
    assert torch.allclose(
        estimate[:2], torch.tensor([truth.x, truth.y], dtype=estimate.dtype), atol=0.03
    )
    expected = torch.tensor(
        [truth.photons, truth.sigma_x, truth.sigma_y, truth.background], dtype=estimate.dtype
    )
    tolerance = torch.tensor([2.0, 0.02, 0.02, 0.02], dtype=estimate.dtype)
    assert torch.all((estimate[2:] - expected).abs() <= tolerance)


def test_preprocessing_method_is_explicit() -> None:
    frames = torch.ones(1, 9, 9)
    assert subtract_background(frames).shape == frames.shape
    with pytest.raises(ValueError, match="method"):
        subtract_background(frames, method="rolling_ball")


def test_all_negative_initialization_has_finite_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def all_negative_iteration(*args: object, **kwargs: object) -> IterationResult:
        empty_candidates = CandidateSet(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
        )
        empty_fit = FitResult(
            torch.empty((0, 6)),
            torch.empty((0, 6)),
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
        )
        return IterationResult(empty_candidates, empty_fit, torch.empty(0, dtype=torch.bool), None)

    monkeypatch.setattr("minutia.pipeline.iterative.run_iteration", all_negative_iteration)
    with pytest.raises(RuntimeError, match="initialization attempts"):
        train_minutia(
            torch.zeros(1, 16, 16),
            config=TrainingConfig(iterations=1, max_initialization_attempts=2),
        )
