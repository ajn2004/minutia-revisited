# MINuTIA Revisited

MINuTIA Revisited is a reproducible reconstruction of the MINuTIA single-
molecule localization method, together with a modern systems design for making
the complete detection-to-localization loop accelerator-resident.

The central idea is different from ordinary object detection:

```text
microscopy frame
      ↓
learned local proposal
      ↓
Gaussian maximum-likelihood localization
      ↓
fit-quality decision
      ↓
positive/negative training example
      ↓
updated detector
```

The detector is not trained from a manually labelled molecule catalogue. A
physical/statistical localizer acts as its label oracle: proposals become
training examples according to whether their localization fit satisfies the
specified tolerances. This fit-guided feedback loop is the defining technical
idea of MINuTIA.

## Why this project matters

The 2018 method demonstrated that a learned detector could adapt through the
success or failure of a downstream localization fit. MINuTIA Revisited
preserves that scientific idea while addressing the engineering bottleneck in
the historical implementation.

The old data path repeatedly crossed the accelerator/host boundary:

```text
detector → full activation image → CPU peak selection → ROI copies → localizer
```

The modern path treats detector proposals as coordinates rather than as a
materialized image and keeps the source frames, candidates, fits, and quality
decisions together on the device:

```text
device frame → detector → threshold/NMS → candidate coordinates → MLE/CRLB
```

This is more than a neural-network rewrite. It is a redesign of the boundary
between computer vision and scientific estimation: sparse detector output is
passed directly to a physics-based estimator, avoiding unnecessary host
synchronization and intermediate ROI transfers. A readable reference path is
retained so that optimization can be checked against scientific correctness.

## What is reproduced

The canonical baseline follows the published 2018 model:

- 7×7 pixel input window (49 features)
- one hidden layer with 30 sigmoid units
- sigmoid output and 0.5 activation threshold
- 5×5 spatial local-maximum selection
- pixel-integrated Gaussian Poisson MLE localization
- Fisher-information/CRLB uncertainty estimates
- configurable fit-quality tolerances
- iterative fit-guided retraining with replay memory

Synthetic data is part of the validation infrastructure, not just a demo. It
provides known molecule positions, photon counts, backgrounds, sub-pixel
locations, noise, and astigmatic PSFs without requiring private laboratory
data. The 2018 reproduction also preserves the historical initialization,
bootstrap, LBFGS training, tolerance oracle, simulation settings, and seeds
where configured.

## Quick start

Install a backend-appropriate PyTorch build first (CPU, CUDA, or ROCm), then:

```bash
uv sync --extra dev --inexact
uv run pytest
```

Run the small seeded 2018 reproduction:

```bash
uv run python -m experiments.reproduce_2018.run \
  --config experiments/reproduce_2018/configs/smoke.toml \
  --output out/reproduce_2018/smoke
```

The run produces machine-readable results, including:

- `results.csv` / `results.json`: per-iteration training and performance
- `identifications.csv`: one audit row per detector candidate, including fit
  parameters, modern and historical quality decisions, and nearest truth
- `matching_sensitivity.csv`: recall and false-identification rates at 0.5,
  1.0, 1.5, and 2.0 px; the canonical configured 1.0 px metric is unchanged
- `ground_truth.json`, `metadata.json`, and a Figure 5-style plot

For a minimal library-level iteration:

```python
from minutia.detector import CanonicalDetector
from minutia.pipeline import run_iteration
from minutia.sim import simulate_movie
from minutia.training import ReplayBuffer

movie = simulate_movie(seed=7)
result = run_iteration(movie.frames, CanonicalDetector(), ReplayBuffer())
print(len(result.candidates.score), result.loss)
```

## Reference and accelerated paths

The project deliberately maintains two levels of implementation:

1. **Reference path** — readable per-candidate localization and explicit
   scientific operations for testing and interpretation.
2. **Batched path** — PyTorch tensor operations for detector, NMS, candidate
   compaction, batched localization, CRLB, and quality evaluation, preserving
   device residency where practical.

The batched path is a portable stepping stone for CPU, CUDA, and ROCm
environments. Custom CUDA/Triton kernels are not treated as automatically
better: they require profiling and numerical-equivalence tests against the
reference implementation.

## Reproduction versus research extensions

The published detector is a single-hidden-layer feed-forward ANN scanned over
image pixels. It is not described as a CNN. Deeper networks, CNN detectors,
alternative PSFs, learned quality models, multi-emitter fitting, replay-policy
changes, and custom fused kernels are experimental extensions and must remain
separate from the canonical reproduction.

## Repository guide

```text
src/minutia/                  scientific and systems implementation
experiments/reproduce_2018/   seeded paper-guided reproduction
tests/                        numerical, device, and end-to-end tests
benchmarks/                   throughput and path comparisons
docs/                         algorithm, architecture, notes, and roadmap
legacy/                       curated MATLAB/CUDA archaeological material
```

Start with:

- [`docs/original_algorithm.md`](docs/original_algorithm.md) — reconstruction
  of the published method
- [`docs/architecture.md`](docs/architecture.md) — historical, reference, and
  device-resident data paths
- [`docs/reproduction_2018.md`](docs/reproduction_2018.md) — experiment design
- [`docs/roadmap.md`](docs/roadmap.md) — implementation status and next steps
- [`legacy/README.md`](legacy/README.md) — provenance and historical divergences

## Citation

Nelson, A. J., & Hess, S. T. (2018). *Molecular imaging with neural training
of identification algorithm (neural network localization identification).*
Microscopy Research and Technique.
<https://doi.org/10.1002/jemt.23059>
