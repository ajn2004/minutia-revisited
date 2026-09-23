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

The reference uses ordinary PyTorch operations and works with normal PyTorch
device semantics on CPU, ROCm, or CUDA. It is intentionally readable rather
than end-to-end device-resident: `localize_candidates` converts candidate
coordinates to Python scalars, loops over candidates, and `_fit_patch` runs a
separate optimizer for each one. Training example assembly and loss reporting
also perform Python scalar conversions. These are documented host
synchronization points, not performance claims.

## Future accelerator-resident design

```text
device frames → detector → device threshold/NMS/compaction
              → (frame, x, y, score) candidates
              → direct source-frame reads → batched MLE/CRLB
              → device labels and training tensors
```

This is an architectural boundary for a later implementation, not the current
behavior. The future path must keep candidate coordinates, localization,
validity checks, and labels as tensors without per-candidate host
synchronization. It must retain a numerical-equivalence test against the
reference path. A compact device gather may be added after profiling, but it
is an optimization rather than a conceptual stage. No custom kernels are part
of this change.

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
