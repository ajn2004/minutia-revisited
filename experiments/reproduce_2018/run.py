"""Run the seeded Figure 5-style fit-guided reproduction.

Example: ``python -m experiments.reproduce_2018.run --config ... --output out``.
"""

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from minutia.detector import CanonicalDetector
from minutia.pipeline.batched import run_iteration_batched
from minutia.pipeline.reference import run_iteration
from minutia.preprocess import subtract_background
from minutia.quality import HistoricalQualityConfig, QualityConfig, quality_oracle
from minutia.training import ReplayBuffer, TensorReplayBuffer

from .config import ExperimentConfig, frames_for_iteration
from .metrics import calculate_metrics, match_truths, matched_rate_metrics
from .plotting import plot_learning_curves
from .simulation import generate_movie


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Observable result of one replay optimization call."""

    final_loss: float
    optimizer_iterations: int


def _empty_training_diagnostics() -> TrainingDiagnostics:
    return TrainingDiagnostics(final_loss=0.0, optimizer_iterations=0)


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


ANALYSIS_MATCHING_RADII = (0.5, 1.0, 1.5, 2.0)
AUDIT_FIELDS = (
    "iteration",
    "source_frame",
    "candidate_x",
    "candidate_y",
    "detector_score",
    "fit_x",
    "fit_y",
    "fit_photons",
    "fit_sigma_x",
    "fit_sigma_y",
    "fit_background",
    "modern_quality_pass",
    "historical_quality_pass",
    "nearest_truth_distance",
    "nearest_truth_photons",
    "nearest_truth_background",
)


def _nearest_truth_audit_values(
    fit_x: float, fit_y: float, source_frame: int, movie: object
) -> tuple[float, float, float]:
    """Find the nearest molecule in the candidate's original source frame."""
    candidates = [m for m in movie.molecules if m.frame == source_frame]
    if not candidates or not (math.isfinite(fit_x) and math.isfinite(fit_y)):
        return float("nan"), float("nan"), float("nan")
    nearest = min(candidates, key=lambda m: (fit_x - m.x) ** 2 + (fit_y - m.y) ** 2)
    distance = ((fit_x - nearest.x) ** 2 + (fit_y - nearest.y) ** 2) ** 0.5
    return float(distance), float(nearest.photons), float(nearest.background)


def _identification_audit_rows(
    iteration: int,
    source_indices: torch.Tensor,
    candidates: torch.Tensor,
    fits: object,
    modern_pass: torch.Tensor,
    historical_pass: torch.Tensor,
    movie: object,
) -> list[dict[str, object]]:
    """Materialize one auditable row for every detector candidate."""
    parameters = fits.parameters.detach().cpu()
    candidate_values = candidates.detach().cpu()
    source_values = source_indices.detach().cpu()
    modern_values = modern_pass.detach().cpu()
    historical_values = historical_pass.detach().cpu()
    rows: list[dict[str, object]] = []
    for index in range(candidate_values.shape[0]):
        candidate = candidate_values[index]
        fit = parameters[index]
        source_frame = int(source_values[int(candidate[0])])
        fit_values = [float(value) for value in fit]
        distance, photons, background = _nearest_truth_audit_values(
            fit_values[0], fit_values[1], source_frame, movie
        )
        rows.append({
            "iteration": iteration,
            "source_frame": source_frame,
            "candidate_x": int(candidate[1]),
            "candidate_y": int(candidate[2]),
            "detector_score": float(candidate[3]),
            "fit_x": fit_values[0],
            "fit_y": fit_values[1],
            "fit_photons": fit_values[2],
            "fit_sigma_x": fit_values[3],
            "fit_sigma_y": fit_values[4],
            "fit_background": fit_values[5],
            "modern_quality_pass": bool(modern_values[index]),
            "historical_quality_pass": bool(historical_values[index]),
            "nearest_truth_distance": distance,
            "nearest_truth_photons": photons,
            "nearest_truth_background": background,
        })
    return rows


def _sensitivity_rows(
    iteration: int,
    truths: list[object],
    identifications: torch.Tensor,
    historical_pass: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    accepted = identifications[historical_pass.to(dtype=torch.bool)]
    for radius in ANALYSIS_MATCHING_RADII:
        detector_rates = matched_rate_metrics(truths, identifications, radius=radius)
        historical_rates = matched_rate_metrics(truths, accepted, radius=radius)
        rows.append({
            "iteration": iteration,
            "radius": radius,
            "detector_recall": detector_rates["recall"],
            "detector_false_identification_fraction": detector_rates[
                "false_identification_fraction"
            ],
            "historical_quality_recall": historical_rates["recall"],
            "historical_quality_false_identification_fraction": historical_rates[
                "false_identification_fraction"
            ],
        })
    return rows


def _sensitivity_result_fields(rows: list[dict[str, object]]) -> dict[str, float]:
    fields: dict[str, float] = {}
    for row in rows:
        radius = str(row["radius"]).replace(".", "_")
        for name in (
            "detector_recall",
            "detector_false_identification_fraction",
            "historical_quality_recall",
            "historical_quality_false_identification_fraction",
        ):
            fields[f"{name}_radius_{radius}"] = float(row[name])
    return fields


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
        count = int(positives.numel() * fraction + 0.5)
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
    # MATLAB's ``round`` is used by Neural_Learning.m for tosspoint.
    count = int(len(positives) * fraction + 0.5)
    if count:
        drop_positions = torch.randperm(len(positives), generator=generator)[:count]
        drop = {positives[int(position)] for position in drop_positions}
        replay.examples = [e for i, e in enumerate(replay.examples) if i not in drop]


def _initialize_detector(detector: CanonicalDetector, config: ExperimentConfig) -> None:
    if config.initialization == "historical_uniform":
        detector.initialize_historical_uniform(config.initialization_epsilon)
    else:
        raise ValueError(f"unknown detector initialization: {config.initialization}")


def _replay_counts(replay: ReplayBuffer | TensorReplayBuffer) -> tuple[int, int]:
    if isinstance(replay, TensorReplayBuffer):
        positives = int(replay.labels.sum())
        return positives, replay.num_examples - positives
    positives = sum(example.label for example in replay.examples)
    return positives, len(replay.examples) - positives


def select_frame_indices(
    n_frames: int, count: int, *, seed: int, iteration: int, attempt: int = 0
) -> torch.Tensor:
    """Select a reproducible, unique random subset of source-frame indices."""
    if not 0 < count <= n_frames:
        raise ValueError(f"count must be in [1, {n_frames}], got {count}")
    generator = torch.Generator().manual_seed(
        seed + 1_000_003 * iteration + 10_007 * attempt
    )
    return torch.randperm(n_frames, generator=generator)[:count]


def _remap_truths(movie: object, source_indices: torch.Tensor) -> list[object]:
    """Return selected truths with frame numbers changed to batch-local indices."""
    source_to_batch = {int(source): batch for batch, source in enumerate(source_indices.tolist())}
    return [
        type(truth)(source_to_batch[truth.frame], truth.x, truth.y, truth.photons,
                    truth.sigma_x, truth.sigma_y, truth.background)
        for truth in movie.molecules
        if truth.frame in source_to_batch
    ]


def _detector_diagnostics(
    frames: torch.Tensor, truths: list[object], detector: CanonicalDetector,
    *, threshold: float,
) -> dict[str, float]:
    """Measure score calibration before thresholding, including truth-centered scores."""
    detector_frames = subtract_background(frames, radius=5, method="rolling_ball_approximation")
    with torch.no_grad():
        scores = detector.score_frames(detector_frames)
        values = scores.reshape(-1)
        truth_scores = torch.stack([
            scores[truth.frame,
                   max(3, min(frames.shape[-2] - 4, round(truth.y))),
                   max(3, min(frames.shape[-1] - 4, round(truth.x)))]
            for truth in truths
        ]) if truths else torch.empty(0, device=frames.device)
    def stat(tensor: torch.Tensor, reducer: str) -> float:
        if not tensor.numel():
            return 0.0
        return float(getattr(tensor, reducer)())
    return {
        "score_min": stat(values, "min"),
        "score_mean": stat(values, "mean"),
        "score_median": stat(values, "median"),
        "score_max": stat(values, "max"),
        "fraction_score_above_threshold": float((values > threshold).to(torch.float32).mean())
        if values.numel() else 0.0,
        "truth_score_mean": stat(truth_scores, "mean"),
        "truth_score_median": stat(truth_scores, "median"),
        "truth_score_min": stat(truth_scores, "min"),
        "truth_score_max": stat(truth_scores, "max"),
    }


def _train_replay(
    detector: CanonicalDetector,
    replay: ReplayBuffer | TensorReplayBuffer,
    *,
    learning_rate: float,
    train_steps: int,
    mode: str = "modern_adam",
    seed: int = 0,
    iteration: int = 0,
) -> TrainingDiagnostics:
    """Train once on accumulated bootstrap examples, after the gate passes."""
    if (mode == "modern_adam" and train_steps <= 0) or (
        replay.num_examples if isinstance(replay, TensorReplayBuffer) else len(replay.examples)
    ) == 0:
        return _empty_training_diagnostics()
    detector.train()
    if isinstance(replay, TensorReplayBuffer):
        patches, target = replay.training_tensors()
    else:
        patches, target = replay.tensors()
    if mode == "modern_adam":
        optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
        loss: torch.Tensor | None = None
        for _ in range(train_steps):
            optimizer.zero_grad()
            prediction = detector(patches)
            loss = nn.functional.binary_cross_entropy(prediction, target)
            loss.backward()
            optimizer.step()
        assert loss is not None
        with torch.no_grad():
            final_loss = nn.functional.binary_cross_entropy(detector(patches), target)
        return TrainingDiagnostics(float(final_loss), train_steps)
    elif mode == "historical_objective_lbfgs":
        generator = torch.Generator(device=patches.device).manual_seed(seed + iteration)
        subset_size = max(1, int(0.9 * patches.shape[0]))
        indices = torch.randperm(
            patches.shape[0], generator=generator, device=patches.device
        )[:subset_size]
        patches, target = patches[indices], target[indices]
        optimizer = torch.optim.LBFGS(
            detector.parameters(), max_iter=100, line_search_fn="strong_wolfe"
        )
        def objective() -> torch.Tensor:
            optimizer.zero_grad()
            prediction = detector(patches)
            loss = nn.functional.binary_cross_entropy(prediction, target)
            loss = loss + 0.3 / (2 * patches.shape[0]) * (
                detector.hidden.weight.square().sum() + detector.output.weight.square().sum()
            )
            loss.backward()
            return loss
        optimizer.step(objective)
        with torch.no_grad():
            prediction = detector(patches)
            loss = nn.functional.binary_cross_entropy(prediction, target)
            loss = loss + 0.3 / (2 * patches.shape[0]) * (
                detector.hidden.weight.square().sum() + detector.output.weight.square().sum()
            )
        # LBFGS stores the actual quasi-Newton iterations in its state.  This
        # is more informative than the number of closure evaluations used by
        # the strong-Wolfe line search.
        optimizer_iterations = max(
            (int(state.get("n_iter", 0)) for state in optimizer.state.values()),
            default=0,
        )
        return TrainingDiagnostics(float(loss), optimizer_iterations)
    else:
        raise ValueError(f"unknown training mode: {mode}")


def _bootstrap(
    config: ExperimentConfig,
    movie: object,
    detector: CanonicalDetector,
    replay: ReplayBuffer | TensorReplayBuffer,
    iteration_runner: object,
    quality: QualityConfig | HistoricalQualityConfig,
) -> tuple[object, list[dict[str, object]]]:
    """Run the historical first-iteration positive-example bootstrap."""
    if config.bootstrap_min_positives < 0:
        raise ValueError("bootstrap_min_positives must be non-negative")
    if config.bootstrap_max_attempts < 1:
        raise ValueError("bootstrap_max_attempts must be positive")
    if isinstance(replay, TensorReplayBuffer):
        replay.clear()
    else:
        replay.examples.clear()
    records: list[dict[str, object]] = []
    last_result: object | None = None
    for attempt in range(1, config.bootstrap_max_attempts + 1):
        _initialize_detector(detector, config)
        count = frames_for_iteration(config.schedule, 1, config.n_frames)
        count = min(count, movie.frames.shape[0])
        selected = select_frame_indices(
            movie.frames.shape[0], count, seed=config.seed, iteration=1, attempt=attempt
        ).to(movie.frames.device)
        attempt_frames = movie.frames[selected]
        last_result = iteration_runner(
            attempt_frames, detector, replay, threshold=config.detector_threshold,
            quality=quality, learning_rate=config.learning_rate, train_steps=0,
            preprocessing_radius=5, minimum_positive_examples=0,
            localization_iterations=config.mle_iterations,
        )
        positives, negatives = _replay_counts(replay)
        record = {
            "bootstrap_attempt": attempt,
            "candidates": int(last_result.candidates.score.shape[0]),
            "positives_this_attempt": int(last_result.labels.sum()),
            "accumulated_positives": positives,
            "accumulated_negatives": negatives,
            "source_frame_indices": selected.tolist(),
        }
        records.append(record)
        print("bootstrap=" + " ".join(f"{key}={value}" for key, value in record.items()))
        if positives >= config.bootstrap_min_positives:
            training = _train_replay(
                detector, replay, learning_rate=config.learning_rate,
                train_steps=config.train_steps,
                mode=config.training_mode, seed=config.seed, iteration=attempt,
            )
            record["final_training_loss"] = training.final_loss
            record["optimizer_iterations"] = training.optimizer_iterations
            return last_result, records
    raise RuntimeError(
        "historical detector bootstrap exhausted "
        f"{config.bootstrap_max_attempts} attempts before reaching "
        f"{config.bootstrap_min_positives} positive examples"
    )


def run(config: ExperimentConfig, output: Path) -> list[dict[str, object]]:
    if config.execution_path not in {"batched", "reference"}:
        raise ValueError("execution_path must be 'batched' or 'reference'")
    if config.training_mode not in {"modern_adam", "historical_objective_lbfgs"}:
        raise ValueError(
            "training_mode must be 'modern_adam' or 'historical_objective_lbfgs'"
        )
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
    if config.quality_mode == "modern":
        quality = QualityConfig()
    elif config.quality_mode == "historical":
        quality = HistoricalQualityConfig()
    else:
        raise ValueError("quality_mode must be 'modern' or 'historical'")
    rows: list[dict[str, object]] = []
    truth_records: list[dict[str, object]] = []
    identification_records: list[dict[str, object]] = []
    sensitivity_records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []
    for iteration in range(1, config.iterations + 1):
        iteration_started = perf_counter()
        training = _empty_training_diagnostics()
        count = frames_for_iteration(config.schedule, iteration, config.n_frames)
        source_indices = select_frame_indices(
            movie.frames.shape[0], count, seed=config.seed, iteration=iteration
        ).to(movie.frames.device)
        frames = movie.frames[source_indices]
        truths = _remap_truths(movie, source_indices)
        if iteration == 1 and not config.bootstrap_enabled:
            _initialize_detector(detector, config)
        detector_diagnostics = _detector_diagnostics(
            frames, truths, detector, threshold=config.detector_threshold
        )
        print(
            f"iteration={iteration} frames={count} candidates=",
            end="",
            flush=True,
        )
        if iteration == 1 and config.bootstrap_enabled:
            result, attempt_records = _bootstrap(
                config, movie, detector, replay, iteration_runner, quality
            )
            bootstrap_records.extend(attempt_records)
            source_indices = torch.tensor(
                attempt_records[-1]["source_frame_indices"], device=movie.frames.device
            )
            frames = movie.frames[source_indices]
            truths = _remap_truths(movie, source_indices)
            detector_diagnostics = _detector_diagnostics(
                frames, truths, detector, threshold=config.detector_threshold
            )
        else:
            result = iteration_runner(
                frames,
                detector,
                replay,
                threshold=config.detector_threshold,
                quality=quality,
                learning_rate=config.learning_rate,
                train_steps=0,
                preprocessing_radius=5,
                minimum_positive_examples=0,
                localization_iterations=config.mle_iterations,
                progress=lambda candidate_count: print(candidate_count, flush=True),
            )
            training_started = perf_counter()
            training = _train_replay(
                detector, replay, learning_rate=config.learning_rate,
                train_steps=config.train_steps, mode=config.training_mode,
                seed=config.seed, iteration=iteration,
            )
            result.timings["replay_training_seconds"] += perf_counter() - training_started
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
        # These flags are analysis outputs only.  In particular, they do not
        # replace ``result.labels`` (the configured training oracle), and the
        # canonical metric above remains at config.matching_radius.
        modern_pass = quality_oracle(result.fits, candidate_tensor, QualityConfig())
        historical_pass = quality_oracle(
            result.fits, candidate_tensor, HistoricalQualityConfig()
        )
        identification_records.extend(
            _identification_audit_rows(
                iteration,
                source_indices,
                candidate_tensor,
                result.fits,
                modern_pass,
                historical_pass,
                movie,
            )
        )
        iteration_sensitivity = _sensitivity_rows(
            iteration, truths, identifications, historical_pass
        )
        sensitivity_records.extend(iteration_sensitivity)
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
                "source_frame": int(source_indices[truth.frame]),
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
            "source_frame_indices": json.dumps(source_indices.tolist()),
            "final_training_loss": training.final_loss,
            "optimizer_iterations": training.optimizer_iterations,
        }
        row.update(_sensitivity_result_fields(iteration_sensitivity))
        if iteration == 1 and config.bootstrap_enabled:
            bootstrap_training = bootstrap_records[-1]
            row["final_training_loss"] = float(bootstrap_training.get("final_training_loss", 0.0))
            row["optimizer_iterations"] = int(bootstrap_training.get("optimizer_iterations", 0))
        row.update(detector_diagnostics)
        positive_count, negative_count = _replay_counts(replay)
        row.update({
            "replay_positive_examples": positive_count,
            "replay_negative_examples": negative_count,
            "replay_positive_fraction": positive_count / (positive_count + negative_count)
            if positive_count + negative_count else 0.0,
            "training_mode": config.training_mode,
        })
        row.update(metrics.as_dict())
        row.update(getattr(result, "timings", {}))
        row.update({
            f"quality_rejected_{name}": count
            for name, count in getattr(result, "quality_rejections", {}).items()
        })
        # The movie is generated once, before iteration one. Repeat the value
        # in every row so each machine-readable iteration record is complete.
        row.setdefault("preprocessing_detection_nms_seconds", 0.0)
        row.setdefault("localization_seconds", 0.0)
        row.setdefault("quality_oracle_seconds", 0.0)
        row.setdefault("replay_training_seconds", 0.0)
        row["total_iteration_seconds"] = perf_counter() - iteration_started
        discard_started = perf_counter()
        # Neural_Learning.m tests ``it - 1 > 10`` after incrementing ``it``;
        # with one-based reporting this starts the toss on iteration 11.
        if iteration >= config.toss_positive_start_iteration:
            _discard_positive_examples(
                replay,
                config.toss_positive_fraction,
                seed=config.seed,
                iteration=iteration,
            )
        row["replay_training_seconds"] = float(row["replay_training_seconds"]) + (
            perf_counter() - discard_started
        )
        positive_count, negative_count = _replay_counts(replay)
        row.update({
            "replay_positive_examples": positive_count,
            "replay_negative_examples": negative_count,
            "replay_positive_fraction": positive_count / (positive_count + negative_count)
            if positive_count + negative_count else 0.0,
        })
        row["total_iteration_seconds"] = perf_counter() - iteration_started
        rows.append(row)
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
        "bootstrap_attempts": bootstrap_records,
        "training_mode": config.training_mode,
        "quality_mode": config.quality_mode,
        "selected_source_frame_indices": [
            json.loads(row["source_frame_indices"]) for row in rows
        ],
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
    with (output / "identifications.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUDIT_FIELDS))
        writer.writeheader()
        writer.writerows(identification_records)
    with (output / "matching_sensitivity.csv").open("w", newline="") as handle:
        sensitivity_fields = (
            "iteration",
            "radius",
            "detector_recall",
            "detector_false_identification_fraction",
            "historical_quality_recall",
            "historical_quality_false_identification_fraction",
        )
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_fields))
        writer.writeheader()
        writer.writerows(sensitivity_records)
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
