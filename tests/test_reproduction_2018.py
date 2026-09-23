import torch

from experiments.reproduce_2018.config import frames_for_iteration
from experiments.reproduce_2018.metrics import calculate_metrics, match_truths
from minutia.sim import Molecule


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


def test_named_historical_schedules() -> None:
    assert frames_for_iteration("paper_methods", 14, 5000) == 10
    assert frames_for_iteration("paper_methods", 15, 5000) == 1000
    assert frames_for_iteration("legacy_matlab", 4, 5000) == 10
    assert frames_for_iteration("legacy_matlab", 5, 5000) == 50
    assert frames_for_iteration("legacy_matlab", 15, 5000) == 500
