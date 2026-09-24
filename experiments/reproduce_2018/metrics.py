"""Metrics and one-to-one truth matching for the 2018 reproduction."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot

import torch

from minutia.sim import Molecule


@dataclass(frozen=True)
class MatchResult:
    """Indices from a deterministic, frame-local one-to-one assignment."""

    matched_truth: tuple[int, ...]
    matched_identifications: tuple[int, ...]
    unmatched_truth: tuple[int, ...]
    unmatched_identifications: tuple[int, ...]


@dataclass(frozen=True)
class Metrics:
    detector_true_positives: int
    detector_false_positives: int
    detector_false_negatives: int
    detector_detection_efficiency: float
    detector_false_identification_fraction: float
    accepted_true_positives: int
    accepted_false_positives: int
    accepted_false_negatives: int
    accepted_detection_efficiency: float
    accepted_false_identification_fraction: float
    total_detector_identifications: int
    identifications_passing_fit_tolerances: int
    known_molecules: int
    identification_precision: float
    fit_success_fraction: float

    # Compatibility aliases for callers that used the original one-stage
    # metrics.  Figure-5 reporting should use the explicitly named accepted
    # (post-tolerance) fields instead.
    @property
    def true_positives(self) -> int:
        return self.detector_true_positives

    @property
    def false_positives(self) -> int:
        return self.detector_false_positives

    @property
    def false_negatives(self) -> int:
        return self.detector_false_negatives

    @property
    def detection_efficiency(self) -> float:
        return self.detector_detection_efficiency

    @property
    def false_identification_fraction(self) -> float:
        return self.detector_false_identification_fraction

    def as_dict(self) -> dict[str, int | float]:
        return {
            "detector_true_positives": self.detector_true_positives,
            "detector_false_positives": self.detector_false_positives,
            "detector_false_negatives": self.detector_false_negatives,
            "detector_detection_efficiency": self.detector_detection_efficiency,
            "detector_false_identification_fraction": self.detector_false_identification_fraction,
            "accepted_true_positives": self.accepted_true_positives,
            "accepted_false_positives": self.accepted_false_positives,
            "accepted_false_negatives": self.accepted_false_negatives,
            "accepted_detection_efficiency": self.accepted_detection_efficiency,
            "accepted_false_identification_fraction": self.accepted_false_identification_fraction,
            "total_detector_identifications": self.total_detector_identifications,
            "identifications_passing_fit_tolerances": self.identifications_passing_fit_tolerances,
            "known_molecules": self.known_molecules,
            "identification_precision": self.identification_precision,
            "fit_success_fraction": self.fit_success_fraction,
        }


def matched_rate_metrics(
    truths: Sequence[Molecule],
    identifications: torch.Tensor,
    *,
    radius: float,
) -> dict[str, float]:
    """Return recall and false-identification fraction for an identification set.

    This small helper is intentionally independent of the canonical metrics
    dataclass.  It is used for the documented matching-radius sensitivity
    analysis and therefore must not alter the configured reproduction metric.
    """
    match = match_truths(truths, identifications, radius=radius)
    true_positives = len(match.matched_identifications)
    total = int(identifications.shape[0])
    return {
        "recall": true_positives / len(truths) if truths else 0.0,
        "false_identification_fraction": (
            (total - true_positives) / total if total else 0.0
        ),
    }


def match_truths(
    truths: Sequence[Molecule],
    identifications: torch.Tensor,
    *,
    radius: float = 1.0,
) -> MatchResult:
    """Match ``(frame, x, y, ...)`` identifications to truths.

    Candidate/truth pairs are sorted by distance, then by their original indices.
    Thus crowded and equal-distance cases have a stable result, and each item can
    occur in at most one pair. Matching is deliberately performed independently
    for each frame.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if identifications.ndim != 2 or identifications.shape[1] < 3:
        raise ValueError("identifications must have columns frame, x, y")
    edges: list[tuple[float, int, int]] = []
    for ti, truth in enumerate(truths):
        for ii in range(identifications.shape[0]):
            row = identifications[ii]
            if int(row[0]) != truth.frame:
                continue
            distance = hypot(float(row[1]) - truth.x, float(row[2]) - truth.y)
            if distance <= radius:
                edges.append((distance, ti, ii))
    used_truth: set[int] = set()
    used_identification: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, ti, ii in sorted(edges):
        if ti not in used_truth and ii not in used_identification:
            used_truth.add(ti)
            used_identification.add(ii)
            pairs.append((ti, ii))
    pairs.sort()
    return MatchResult(
        tuple(ti for ti, _ in pairs),
        tuple(ii for _, ii in pairs),
        tuple(i for i in range(len(truths)) if i not in used_truth),
        tuple(i for i in range(identifications.shape[0]) if i not in used_identification),
    )


def calculate_metrics(
    truths: Sequence[Molecule],
    identifications: torch.Tensor,
    passing: torch.Tensor,
    *,
    matching_radius: float = 1.0,
) -> tuple[Metrics, MatchResult]:
    """Calculate explicit Figure-5-style rates and return the assignment."""
    if passing.numel() != identifications.shape[0]:
        raise ValueError("one fit-quality flag is required per identification")
    detector_match = match_truths(truths, identifications, radius=matching_radius)
    detector_tp = len(detector_match.matched_identifications)
    total = int(identifications.shape[0])
    detector_fp = total - detector_tp
    known = len(truths)
    detector_fn = known - detector_tp
    accepted = identifications[passing.to(dtype=torch.bool)]
    # This is deliberately a new assignment, rather than filtering the
    # pre-tolerance assignment: acceptance can change which identification is
    # the closest valid match in crowded scenes.
    accepted_match = match_truths(truths, accepted, radius=matching_radius)
    accepted_tp = len(accepted_match.matched_identifications)
    accepted_total = int(accepted.shape[0])
    accepted_fp = accepted_total - accepted_tp
    accepted_fn = known - accepted_tp
    passing_count = accepted_total
    detector_denominator = detector_tp + detector_fp
    accepted_denominator = accepted_tp + accepted_fp
    return Metrics(
        detector_tp,
        detector_fp,
        detector_fn,
        detector_tp / known if known else 0.0,
        detector_fp / detector_denominator if detector_denominator else 0.0,
        accepted_tp,
        accepted_fp,
        accepted_fn,
        accepted_tp / known if known else 0.0,
        accepted_fp / accepted_denominator if accepted_denominator else 0.0,
        total,
        passing_count,
        known,
        detector_tp / detector_denominator if detector_denominator else 0.0,
        passing_count / total if total else 0.0,
    ), detector_match
