# MINuTIA Revisited

```text
raw frame → detector score → candidate → MLE fit
          → accepted/rejected training label → improved detector
```

MINuTIA is unusual because its detector is trained by the downstream physical
localizer: a proposed image region becomes a positive or negative example
according to whether its fit satisfies explicit quality tolerances. This
repository provides a reproducible Python/PyTorch reconstruction of the 2018
method, preserves historical MATLAB/CUDA sources as archaeology, and develops
a device-resident detector-to-localizer redesign.

## What is implemented

The canonical baseline is a 7×7 single-hidden-layer sigmoid MLP with 30 hidden
units, 0.5 thresholding, 5×5 local-maximum suppression, pixel-integrated
Gaussian Poisson fitting, Fisher/CRLB uncertainty, and fit-guided replay
training. Synthetic ordinary and astigmatic PSFs are first-class validation
data, so no private microscopy files are required.

The modern fast-path contract is `localize_candidates(frames, candidates)`:
candidate tensors contain frame, x, y, and detector score and remain on the
same device as the source frames. The current implementation is a clear
reference path; CUDA profiling and kernel fusion intentionally come later.

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

The core modules are organized under `src/minutia/{sim,preprocess,detector,
localization,quality,training,pipeline}`. A minimal experiment is:

```python
from minutia.detector import CanonicalDetector
from minutia.pipeline import run_iteration
from minutia.sim import simulate_movie
from minutia.training import ReplayBuffer

movie = simulate_movie(seed=7)
result = run_iteration(movie.frames, CanonicalDetector(), ReplayBuffer())
print(len(result.candidates.score), result.loss)
```

## Reproduction versus extensions

The published model is not described as a CNN: it is a feed-forward ANN
scanned over image pixels. Deeper networks, CNNs, alternative PSFs, replay
policies, and custom kernels are extensions and will be labelled as such.
Historical architecture drift is documented in [`legacy/README.md`](legacy/README.md).
See [`docs/original_algorithm.md`](docs/original_algorithm.md),
[`docs/architecture.md`](docs/architecture.md), and
[`docs/roadmap.md`](docs/roadmap.md) for scientific and systems details.

## Citation

Nelson, A. J., & Hess, S. T. (2018). *Molecular imaging with neural training
of identification algorithm (neural network localization identification).* 
Microscopy Research and Technique. https://doi.org/10.1002/jemt.23059
