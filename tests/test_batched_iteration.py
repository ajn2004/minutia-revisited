import torch

from minutia.detector import CanonicalDetector
from minutia.localization import localize_candidates_batched
from minutia.pipeline.batched import run_iteration_batched
from minutia.pipeline.reference import run_iteration
from minutia.sim import simulate_movie
from minutia.training import ReplayBuffer, TensorReplayBuffer


def test_tensor_replay_retention_is_tensor_native() -> None:
    replay = TensorReplayBuffer(capacity=2, device="cpu", seed=4)
    patches = torch.arange(3 * 49, dtype=torch.float32).reshape(3, 7, 7)
    replay.add(
        patches,
        torch.tensor([True, False, True]),
        torch.tensor([0, 0, 0]),
        torch.tensor([3, 4, 5]),
        torch.tensor([3, 4, 5]),
        torch.ones(3),
        torch.zeros(3, 6),
    )
    assert replay.num_examples == 2
    assert all(getattr(replay, name).device.type == "cpu" for name in replay._fields)
    assert replay.patches.shape == (2, 7, 7)


def test_batched_iteration_matches_reference_iteration() -> None:
    movie = simulate_movie(
        n_frames=2,
        height=32,
        width=32,
        molecules_per_frame=2,
        photons=1600.0,
        background=4.0,
        poisson=False,
        seed=21,
    )
    reference_detector = CanonicalDetector()
    batched_detector = CanonicalDetector()
    batched_detector.load_state_dict(reference_detector.state_dict())
    reference_replay = ReplayBuffer(capacity=500)
    batched_replay = TensorReplayBuffer(capacity=500, device=movie.frames.device)
    kwargs = dict(
        threshold=0.5,
        preprocess=False,
        train_steps=1,
        learning_rate=1e-3,
    )
    frames = movie.frames.to(torch.float32)
    reference = run_iteration(frames, reference_detector, reference_replay, **kwargs)
    batched = run_iteration_batched(
        frames, batched_detector, batched_replay, **kwargs
    )
    assert torch.equal(reference.candidates.as_tensor(), batched.candidates.as_tensor())
    assert torch.allclose(
        reference.fits.parameters,
        batched.fits.parameters,
        atol=0.25,
        rtol=0.05,
        equal_nan=True,
    )
    assert torch.equal(reference.labels, batched.labels)
    assert torch.equal(
        batched_replay.frame, torch.tensor([e.frame for e in reference_replay.examples])
    )
    assert torch.equal(
        batched_replay.x, torch.tensor([e.x for e in reference_replay.examples])
    )
    assert torch.equal(
        batched_replay.y, torch.tensor([e.y for e in reference_replay.examples])
    )
    assert torch.equal(
        batched_replay.labels, torch.tensor([e.label for e in reference_replay.examples])
    )
    assert torch.allclose(
        batched_replay.fit_parameters,
        torch.stack([e.fit_parameters for e in reference_replay.examples]),
        equal_nan=True,
    )
    for expected, actual in zip(reference_detector.parameters(), batched_detector.parameters()):
        assert torch.allclose(expected, actual, atol=0.25, rtol=0.05)


def test_batched_iteration_keeps_reporting_values_as_tensors() -> None:
    frames = torch.ones(1, 24, 24)
    detector = CanonicalDetector()
    replay = TensorReplayBuffer(device=frames.device)
    result = run_iteration_batched(frames, detector, replay, preprocess=False, train_steps=0)
    assert result.loss.device == frames.device
    assert result.positive_count.device == frames.device
    assert result.candidate_count.device == frames.device


def test_localizer_chunking_preserves_batched_results() -> None:
    frames = torch.ones(2, 24, 24, dtype=torch.float64)
    candidates = torch.tensor(
        [[0, 12, 12, 1.0], [1, 13, 12, 0.9], [0, 11, 13, 0.8]], dtype=torch.float64
    )
    whole = localize_candidates_batched(frames, candidates, iterations=2)
    chunked = localize_candidates_batched(frames, candidates, iterations=2, chunk_size=2)
    assert torch.allclose(whole.parameters, chunked.parameters, equal_nan=True)
    assert torch.equal(whole.valid, chunked.valid)
