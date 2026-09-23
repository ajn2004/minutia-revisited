from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.reproduce_2018.config import ExperimentConfig, frames_for_iteration
from experiments.reproduce_2018.metrics import calculate_metrics, match_truths
from experiments.reproduce_2018.run import run
from minutia.detector import CanonicalDetector
from minutia.localization import FitResult
from minutia.quality import HistoricalQualityConfig, quality_oracle, quality_rejection_counts
from minutia.sim import Molecule
from minutia.training import ReplayBuffer, TrainingExample


def molecule(frame: int, x: float, y: float) -> Molecule:
    return Molecule(frame, x, y, 200.0, 1.2, 1.2, 4.0)


def test_matching_is_one_to_one_and_frame_local() -> None:
    truths = [molecule(0, 5, 5), molecule(0, 5.8, 5), molecule(1, 5, 5)]
    ids = torch.tensor([[0, 5.2, 5.0, 0.8], [0, 5.3, 5.0, 0.7], [1, 5.0, 5.0, 0.9]])
    result = match_truths(truths, ids, radius=0.5)
    assert result.matched_truth == (0, 2)
    assert result.matched_identifications == (0, 2)
    assert result.unmatched_truth == (1,)
    assert result.unmatched_identifications == (1,)


def test_metrics_use_explicit_denominators() -> None:
    truths = [molecule(0, 5, 5), molecule(0, 10, 10)]
    ids = torch.tensor([[0, 5, 5, 0.9], [0, 2, 2, 0.8]])
    metrics, _ = calculate_metrics(truths, ids, torch.tensor([True, False]), matching_radius=0.5)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.identification_precision == 0.5
    assert metrics.detection_efficiency == 0.5
    assert metrics.fit_success_fraction == 0.5
    assert metrics.accepted_true_positives == 1
    assert metrics.accepted_false_positives == 0
    assert metrics.accepted_false_negatives == 1
    assert metrics.accepted_detection_efficiency == 0.5


def test_named_historical_schedules() -> None:
    assert frames_for_iteration("paper_methods", 14, 5000) == 10
    assert frames_for_iteration("paper_methods", 15, 5000) == 1000
    assert frames_for_iteration("legacy_matlab", 4, 5000) == 10
    assert frames_for_iteration("legacy_matlab", 5, 5000) == 50
    assert frames_for_iteration("legacy_matlab", 15, 5000) == 500


def test_historical_quality_uses_covariance_variances_and_strict_gates() -> None:
    parameters = torch.tensor([[10.2, 20.2, 100.0, 2.0, 2.0, 4.0]])
    covariance = torch.diag(torch.tensor([0.2, 0.2, 100.0, 1.0, 1.0, 1.0])).unsqueeze(0)
    # Deliberately disagree with covariance: historical quality must not use
    # this standard-deviation tensor for the CRLB variance gates.
    uncertainty = torch.full_like(parameters, 100.0)
    fits = FitResult(
        parameters,
        uncertainty,
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        covariance=covariance,
    )
    candidates = torch.tensor([[0.0, 10.0, 20.0, 1.0]])

    assert bool(quality_oracle(fits, candidates, HistoricalQualityConfig())[0])

    at_boundary = parameters.clone()
    at_boundary[0, 2] = 10.0
    boundary_fits = FitResult(
        at_boundary, uncertainty, torch.zeros(1), torch.ones(1, dtype=torch.bool),
        covariance=covariance,
    )
    assert not bool(quality_oracle(boundary_fits, candidates, HistoricalQualityConfig())[0])
    counts = quality_rejection_counts(boundary_fits, candidates, HistoricalQualityConfig())
    assert counts["photons"] == 1


def test_experiment_defaults_preserve_historical_positive_toss_start() -> None:
    config = ExperimentConfig.from_toml("experiments/reproduce_2018/configs/smoke.toml")
    assert config.quality_mode == "modern"
    assert config.toss_positive_start_iteration == 11


def test_smoke_reproduction_uses_batched_path(tmp_path: Path) -> None:
    config = ExperimentConfig.from_toml("experiments/reproduce_2018/configs/smoke.toml")
    rows = run(config, tmp_path / "smoke")
    assert len(rows) == config.iterations
    assert rows[0]["candidate_count"] >= 0
    for key in (
        "detector_true_positives",
        "detector_false_positives",
        "detector_false_negatives",
        "detector_detection_efficiency",
        "detector_false_identification_fraction",
        "accepted_true_positives",
        "accepted_false_positives",
        "accepted_false_negatives",
        "accepted_detection_efficiency",
        "accepted_false_identification_fraction",
        "fit_success_fraction",
        "simulation_seconds",
        "preprocessing_detection_nms_seconds",
        "localization_seconds",
        "quality_oracle_seconds",
        "truth_matching_metrics_seconds",
        "replay_training_seconds",
        "total_iteration_seconds",
    ):
        assert key in rows[0]
    assert '"execution_path": "batched"' in (tmp_path / "smoke" / "metadata.json").read_text()


def test_historical_initialization_and_bootstrap_gate(monkeypatch) -> None:
    """The reproduction must not optimize an all-negative bootstrap set."""
    detector = CanonicalDetector()
    detector.initialize_historical_uniform()
    assert all(
        bool(torch.all(parameter <= 0.12) and torch.all(parameter >= -0.12))
        for parameter in detector.parameters()
    )

    from experiments.reproduce_2018 import run as reproduction

    original_train = reproduction._train_replay
    train_calls: list[int] = []

    def record_train(*args, **kwargs):
        train_calls.append(1)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(reproduction, "_train_replay", record_train)
    config = ExperimentConfig.from_toml("experiments/reproduce_2018/configs/smoke.toml")
    config = config.__class__(**{
        **config.__dict__, "bootstrap_min_positives": 1, "bootstrap_max_attempts": 2
    })
    replay = ReplayBuffer()
    calls = 0

    def fake_iteration(frames, _detector, replay, **kwargs):
        nonlocal calls
        calls += 1
        label = calls == 2
        replay.add([TrainingExample(torch.zeros(7, 7), label, 0, 3, 3, 0.5,
                                    torch.zeros(6))])
        return SimpleNamespace(
            candidates=SimpleNamespace(score=torch.ones(1)),
            labels=torch.tensor([label]),
        )

    movie = SimpleNamespace(frames=torch.zeros(2, 16, 16))
    detector = CanonicalDetector()
    result, attempts = reproduction._bootstrap(
        config, movie, detector, replay, fake_iteration, reproduction.QualityConfig()
    )
    assert result is not None
    assert [record["accumulated_positives"] for record in attempts] == [0, 1]
    assert len(train_calls) == 1  # no optimizer call occurred during attempt 1
