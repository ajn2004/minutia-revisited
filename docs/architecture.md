# Architecture

## Historical path

```text
frame → rolling-ball subtraction → GPU ANN activation image
      → host peak selection/segmentation → copied ROI stack
      → GPU MLE/CRLB → host tolerances → MATLAB training arrays
```

## Faithful reference path

```text
frames → preprocessing → 7×7/30 ANN → 5×5 NMS
       → candidate coordinates → pixel-integrated Gaussian MLE
       → Fisher/CRLB → tolerance oracle → replay buffer → ANN training
```

The readable reference uses ordinary PyTorch operations and works with normal
PyTorch device semantics on CPU, ROCm, or CUDA. It is intentionally readable
rather than accelerator-friendly: `localize_candidates` converts candidate
coordinates to Python scalars, loops over candidates, and `_fit_patch` runs a
separate optimizer for each one. It remains the numerical oracle.

The portable batched path is exposed separately as
`localize_candidates_batched`. It gathers `[N,K,K]` windows with tensor
indexing, runs a damped Newton/Fisher update over an `[N,6]` parameter tensor,
and returns tensor-valued parameters, uncertainty, Fisher information,
covariance, likelihood, and validity masks. It uses the same pixel-integrated
Gaussian model and supports CPU, CUDA, and ROCm through PyTorch's normal
`torch.cuda` device API. `chunk_size` bounds the temporary candidate working
set for large batches; it is configurable rather than tuned to a synthetic
benchmark.

## Tensor-native iteration

`run_iteration_batched` is the accelerator-resident/tensor-native path:

```text
device frames → preprocessing → CanonicalDetector → 5×5 NMS
              → device candidate tensor → batched MLE/CRLB
              → tensor quality labels → batched 7×7 gather
              → TensorReplayBuffer → detector training
```

The `TensorReplayBuffer` stores patches, labels, coordinates, scores, and fit
parameters as tensors on one configured device. Capacity retention uses tensor
indexing. The Python-object `ReplayBuffer` and `run_iteration` remain the
readable scientific/debugging oracle. This path is accelerator-resident, not
fused: it uses ordinary portable PyTorch operations and no custom kernels.

## Device-resident detector → localizer boundary (not fused)

```text
device frames → detector → device threshold/NMS/compaction
              → (frame, x, y, score) candidates
              → direct source-frame reads → batched MLE/CRLB
              → device labels and training tensors
```

The batched localizer and tensor-native iteration now provide this portable
device-resident path. Candidate coordinates, localization validity checks,
labels, replay retention, and training patches remain tensors without
per-candidate host synchronization. A compact device gather is an
implementation optimization rather than a public architectural boundary. No
custom kernels are part of this change.

`benchmarks/benchmark_localization.py` reports elapsed time for the reference
and batched paths on CPU and, when available, an accelerator. The complete
path is measured by `benchmarks/benchmark_pipeline.py`; run it before making
performance claims.

## Preprocessing boundary

`run_iteration` explicitly preprocesses frames for detector scoring. The
default `rolling_ball_approximation` name identifies the paper's rolling-ball
method as the canonical target while making clear that the current portable
implementation is a local-mean approximation. Localization reads the original
frames so its Poisson model retains the acquisition background. The method and
radius are configurable; a calibrated rolling-ball implementation remains
future work.

## Installation and accelerator backends

PyTorch is deliberately not pinned in the project dependency or lock file.
This prevents a generic lock resolver from replacing an existing ROCm build
with a CUDA build (or vice versa). Install the appropriate PyTorch distribution
first, then install this project and its development tools:

```bash
# CPU: use the CPU command for the selected release from pytorch.org
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# NVIDIA CUDA: use the matching command/index from pytorch.org
python -m pip install torch
# AMD ROCm: use the matching ROCm command/index from pytorch.org
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm<version>
# --inexact preserves the separately installed backend-specific PyTorch.
uv sync --extra dev --inexact
```

The application uses backend-neutral `torch.device` and tensor operations;
backend selection belongs to the environment installation.
