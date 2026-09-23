"""Run the seeded Figure 5-style fit-guided reproduction.

Example: ``python -m experiments.reproduce_2018.run --config ... --output out``.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import torch

from minutia.detector import CanonicalDetector
from minutia.pipeline.reference import run_iteration
from minutia.quality import QualityConfig
from minutia.training import ReplayBuffer

from .config import ExperimentConfig, frames_for_iteration
from .metrics import calculate_metrics
from .plotting import plot_learning_curves
from .simulation import generate_movie


def _mean_photons(molecules: list[object]) -> float:
    return sum(m.photons for m in molecules) / len(molecules) if molecules else 0.0


def _median_photons(molecules: list[object]) -> float:
    return float(torch.tensor([m.photons for m in molecules]).median()) if molecules else 0.0


def _mean_signal_proxy(molecules: list[object]) -> float:
    return (
        sum(m.photons / (m.background**0.5) for m in molecules) / len(molecules)
        if molecules
        else 0.0
    )


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(config: ExperimentConfig, output: Path) -> list[dict[str, object]]:
    torch.manual_seed(config.seed)
    movie = generate_movie(config)
    detector = CanonicalDetector()
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
    quality = QualityConfig()
    rows: list[dict[str, object]] = []
    truth_records: list[dict[str, object]] = []
    for iteration in range(1, config.iterations + 1):
        count = frames_for_iteration(config.schedule, iteration, config.n_frames)
        frames = movie.frames[:count]
        truths = [m for m in movie.molecules if m.frame < count]
        result = run_iteration(
            frames,
            detector,
            replay,
            threshold=config.detector_threshold,
            quality=quality,
            learning_rate=config.learning_rate,
            train_steps=config.train_steps,
            preprocessing_radius=5,
            minimum_positive_examples=0,
            localization_iterations=config.mle_iterations,
        )
        candidate_tensor = result.candidates.as_tensor()
        # Fit parameters contain image x/y; retain the candidate frame index.
        identifications = torch.cat((candidate_tensor[:, :1], result.fits.parameters[:, :2]), dim=1)
        metrics, match = calculate_metrics(
            truths,
            identifications,
            result.labels,
            matching_radius=config.matching_radius,
        )
        detected = [truths[i] for i in match.matched_truth]
        missed = [truths[i] for i in match.unmatched_truth]
        detected_truth_indices = set(match.matched_truth)
        truth_records.extend(
            {
                "regime": config.name,
                "iteration": iteration,
                "frame": truth.frame,
                "photon_count": truth.photons,
                "background": truth.background,
                "photon_over_sqrt_background": truth.photons / (truth.background**0.5),
                "detected": index in detected_truth_indices,
            }
            for index, truth in enumerate(truths)
        )
        row: dict[str, object] = {
            "regime": config.name,
            "iteration": iteration,
            "frames_analyzed": count,
            "training_examples": len(replay.examples),
            "candidate_count": len(result.candidates.score),
            "mean_photons_detected": _mean_photons(detected),
            "median_photons_detected": _median_photons(detected),
            "mean_photons_missed": _mean_photons(missed),
            "median_photons_missed": _median_photons(missed),
            "mean_photons_sqrt_background_detected": _mean_signal_proxy(detected),
            "mean_photons_sqrt_background_missed": _mean_signal_proxy(missed),
            "mean_photons_sqrt_offset_detected": _mean_signal_proxy(detected),
            "mean_photons_sqrt_offset_missed": _mean_signal_proxy(missed),
        }
        row.update(metrics.as_dict())
        rows.append(row)
        if config.toss_positive_fraction:
            positives = [i for i, example in enumerate(replay.examples) if example.label]
            keep = int(len(positives) * config.toss_positive_fraction)
            if keep:
                generator = torch.Generator().manual_seed(config.seed + iteration)
                drop_positions = torch.randperm(len(positives), generator=generator)[:keep]
                drop = {positives[int(position)] for position in drop_positions}
                replay.examples = [e for i, e in enumerate(replay.examples) if i not in drop]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.__dict__,
        "seed": config.seed,
        "git_commit": _commit(),
        "torch_version": torch.__version__,
        "device": str(movie.frames.device),
        "backend": "cpu",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (output / "ground_truth.json").write_text(
        json.dumps([molecule.__dict__ for molecule in movie.molecules], indent=2)
    )
    (output / "truth_classification.json").write_text(json.dumps(truth_records, indent=2))
    with (output / "results.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_learning_curves(output / "results.csv", output / "figure5_style.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(ExperimentConfig.from_toml(args.config), args.output)


if __name__ == "__main__":
    main()
