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
    true_positives: int
    false_positives: int
    false_negatives: int
    total_detector_identifications: int
    identifications_passing_fit_tolerances: int
    known_molecules: int
    identification_precision: float
    false_identification_fraction: float
    detection_efficiency: float
    fit_success_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_detector_identifications": self.total_detector_identifications,
            "identifications_passing_fit_tolerances": self.identifications_passing_fit_tolerances,
            "known_molecules": self.known_molecules,
            "identification_precision": self.identification_precision,
            "false_identification_fraction": self.false_identification_fraction,
            "detection_efficiency": self.detection_efficiency,
            "fit_success_fraction": self.fit_success_fraction,
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
    match = match_truths(truths, identifications, radius=matching_radius)
    tp = len(match.matched_identifications)
    total = int(identifications.shape[0])
    fp = total - tp
    known = len(truths)
    fn = known - tp
    passing_count = int(passing.to(torch.int64).sum())
    denominator = tp + fp
    return Metrics(
        tp,
        fp,
        fn,
        total,
        passing_count,
        known,
        tp / denominator if denominator else 0.0,
        fp / denominator if denominator else 0.0,
        tp / known if known else 0.0,
        passing_count / total if total else 0.0,
    ), match
