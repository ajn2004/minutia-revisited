"""Run the seeded Figure 5-style fit-guided reproduction.

Example: ``python -m experiments.reproduce_2018.run --config ... --output out``.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from time import perf_counter

import torch

from minutia.detector import CanonicalDetector
from minutia.pipeline.batched import run_iteration_batched
from minutia.pipeline.reference import run_iteration
from minutia.quality import QualityConfig
from minutia.training import ReplayBuffer, TensorReplayBuffer

from .config import ExperimentConfig, frames_for_iteration
from .metrics import calculate_metrics, match_truths
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


def _discard_positive_examples(replay: ReplayBuffer | TensorReplayBuffer, fraction: float,
                               *, seed: int, iteration: int) -> None:
    """Apply the historical post-iteration positive-example policy.

    ``fraction`` is the fraction discarded (the configuration name is retained
    for compatibility with the historical reproduction).  This is deliberately
    separate from each replay implementation's generic capacity policy.
    """
    if not fraction:
        return
    generator_device = replay.device if isinstance(replay, TensorReplayBuffer) else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed + iteration)
    if isinstance(replay, TensorReplayBuffer):
        positives = torch.where(replay.labels)[0]
        count = int(positives.numel() * fraction)
        if count:
            permutation = torch.randperm(
                positives.numel(), generator=generator, device=positives.device
            )
            drop = positives[permutation[:count]]
            keep = torch.ones(replay.num_examples, dtype=torch.bool, device=replay.device)
            keep[drop] = False
            replay.retain_indices(torch.where(keep)[0])
        return
    positives = [i for i, example in enumerate(replay.examples) if example.label]
    count = int(len(positives) * fraction)
    if count:
        drop_positions = torch.randperm(len(positives), generator=generator)[:count]
        drop = {positives[int(position)] for position in drop_positions}
        replay.examples = [e for i, e in enumerate(replay.examples) if i not in drop]


def run(config: ExperimentConfig, output: Path) -> list[dict[str, object]]:
    if config.execution_path not in {"batched", "reference"}:
        raise ValueError("execution_path must be 'batched' or 'reference'")
    torch.manual_seed(config.seed)
    simulation_started = perf_counter()
    movie = generate_movie(config)
    simulation_seconds = perf_counter() - simulation_started
    detector = CanonicalDetector().to(
        device=movie.frames.device,
        dtype=movie.frames.dtype,
    )
    if config.execution_path == "batched":
        replay = TensorReplayBuffer(
            capacity=config.replay_capacity,
            device=movie.frames.device,
            seed=config.seed,
        )
        iteration_runner = run_iteration_batched
    else:
        replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
        iteration_runner = run_iteration
    quality = QualityConfig()
    rows: list[dict[str, object]] = []
    truth_records: list[dict[str, object]] = []
    for iteration in range(1, config.iterations + 1):
        iteration_started = perf_counter()
        count = frames_for_iteration(config.schedule, iteration, config.n_frames)
        frames = movie.frames[:count]
        truths = [m for m in movie.molecules if m.frame < count]
        print(
            f"iteration={iteration} frames={count} candidates=",
            end="",
            flush=True,
        )
        result = iteration_runner(
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
            progress=lambda candidate_count: print(candidate_count, flush=True),
        )
        matching_started = perf_counter()
        candidate_tensor = result.candidates.as_tensor()
        # Fit parameters contain image x/y; retain the candidate frame index.
        identifications = torch.cat(
            (candidate_tensor[:, :1], result.fits.parameters[:, :2]), dim=1
        ).detach()
        metrics, match = calculate_metrics(
            truths,
            identifications,
            result.labels,
            matching_radius=config.matching_radius,
        )
        matching_metrics_seconds = perf_counter() - matching_started
        accepted_identifications = identifications[result.labels.to(dtype=torch.bool)]
        accepted_match = match_truths(
            truths, accepted_identifications, radius=config.matching_radius
        )
        detected = [truths[i] for i in match.matched_truth]
        missed = [truths[i] for i in match.unmatched_truth]
        detected_truth_indices = set(match.matched_truth)
        accepted_truth_indices = set(accepted_match.matched_truth)
        truth_records.extend(
            {
                "regime": config.name,
                "iteration": iteration,
                "frame": truth.frame,
                "photon_count": truth.photons,
                "background": truth.background,
                "photon_over_sqrt_background": truth.photons / (truth.background**0.5),
                "detected": index in detected_truth_indices,
                "accepted": index in accepted_truth_indices,
            }
            for index, truth in enumerate(truths)
        )
        row: dict[str, object] = {
            "regime": config.name,
            "iteration": iteration,
            "frames_analyzed": count,
            "training_examples": (
                replay.num_examples
                if isinstance(replay, TensorReplayBuffer)
                else len(replay.examples)
            ),
            "candidate_count": len(result.candidates.score),
            "mean_photons_detected": _mean_photons(detected),
            "median_photons_detected": _median_photons(detected),
            "mean_photons_missed": _mean_photons(missed),
            "median_photons_missed": _median_photons(missed),
            "mean_photons_sqrt_background_detected": _mean_signal_proxy(detected),
            "mean_photons_sqrt_background_missed": _mean_signal_proxy(missed),
            "mean_photons_sqrt_offset_detected": _mean_signal_proxy(detected),
            "mean_photons_sqrt_offset_missed": _mean_signal_proxy(missed),
            "simulation_seconds": simulation_seconds,
            "truth_matching_metrics_seconds": matching_metrics_seconds,
        }
        row.update(metrics.as_dict())
        row.update(getattr(result, "timings", {}))
        # The movie is generated once, before iteration one. Repeat the value
        # in every row so each machine-readable iteration record is complete.
        row.setdefault("preprocessing_detection_nms_seconds", 0.0)
        row.setdefault("localization_seconds", 0.0)
        row.setdefault("quality_oracle_seconds", 0.0)
        row.setdefault("replay_training_seconds", 0.0)
        row["total_iteration_seconds"] = perf_counter() - iteration_started
        rows.append(row)
        discard_started = perf_counter()
        _discard_positive_examples(
            replay,
            config.toss_positive_fraction,
            seed=config.seed,
            iteration=iteration,
        )
        row["replay_training_seconds"] = float(row["replay_training_seconds"]) + (
            perf_counter() - discard_started
        )
        row["total_iteration_seconds"] = perf_counter() - iteration_started
        print(
            f"iteration={iteration} detector_tp={metrics.detector_true_positives} "
            f"detector_fp={metrics.detector_false_positives} "
            f"detector_fn={metrics.detector_false_negatives} "
            f"accepted_tp={metrics.accepted_true_positives} "
            f"accepted_fp={metrics.accepted_false_positives} "
            f"accepted_fn={metrics.accepted_false_negatives} "
            f"fit_success_fraction={metrics.fit_success_fraction}",
            flush=True,
        )
        print(
            "timing_seconds="
            + " ".join(
                f"{name.removesuffix('_seconds')}={float(row[name]):.6f}"
                for name in (
                    "simulation_seconds",
                    "preprocessing_detection_nms_seconds",
                    "localization_seconds",
                    "quality_oracle_seconds",
                    "truth_matching_metrics_seconds",
                    "replay_training_seconds",
                    "total_iteration_seconds",
                )
            ),
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config.__dict__,
        "seed": config.seed,
        "git_commit": _commit(),
        "torch_version": torch.__version__,
        "device": str(movie.frames.device),
        "backend": "cpu",
        "execution_path": config.execution_path,
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
