# AGENTS.md

# MINuTIA Revisited — Agent Guide

## 1. Project Mission

MINuTIA Revisited is a modern, public, reproducible reimplementation and extension of the algorithm described in:

> A. J. Nelson and S. T. Hess, “Molecular imaging with neural training of identification algorithm (neural network localization identification),” *Microscopy Research and Technique* (2018). DOI: 10.1002/jemt.23059.

The project has three goals:

1. **Faithfully reproduce the published MINuTIA method.**
2. **Modernize the software architecture and accelerator path without changing the scientific intent.**
3. **Provide a clean research/engineering artifact that demonstrates computer vision, scientific machine learning, numerical optimization, GPU systems work, testing, and reproducibility.**

This is not a line-by-line MATLAB port. Historical MATLAB/CUDA code is archaeological reference material. Reconstruct the intended algorithm from the paper plus legacy code, document inconsistencies, and implement a clean modern system.

---

## 2. Source-of-Truth Order

When sources disagree, use the following precedence:

1. **Published paper** — canonical scientific description.
2. **Consistent historical implementation** — evidence of how the algorithm was actually executed.
3. **Documented engineering judgment** — only when the first two are insufficient.

Do not silently reconcile disagreements.

If the paper and legacy implementation differ, document the disagreement in `legacy/README.md` and, where relevant, in `docs/original_algorithm.md`.

Never invent historical behavior and present it as canonical.

---

## 3. What MINuTIA Is Actually Learning

The detector is **not simply learning “is this a molecule?” from human labels**.

The core learning target is closer to:

> **Will the image region around this pixel produce a localization fit that satisfies the fit-quality criteria?**

The downstream localization algorithm acts as a label oracle.

Conceptually:

```text
detector proposes region
        ↓
physical/statistical localization fit
        ↓
fit quality evaluation
        ↓
positive or negative training label
        ↓
detector retraining
```

The feedback loop between learned detection and downstream fit success is the defining feature of MINuTIA.

Do not replace this with an ordinary supervised object-classification pipeline and still call it the canonical MINuTIA implementation.

---

## 4. Canonical Published Pipeline

The original method should be represented as:

```text
raw microscopy frames
        ↓
background preprocessing
        ↓
ANN evaluates local image regions around each pixel
        ↓
activation threshold
        ↓
spatial local-maximum selection
        ↓
candidate region segmentation
        ↓
MLE localization fit
        ↓
fit-quality / tolerance evaluation
        ↓
positive and negative examples
        ↓
ANN retraining
        ↓
repeat on randomized files/frames
```

The published implementation uses rolling-ball subtraction for background preprocessing, evaluates each pixel independently, localizes identified regions with a Gaussian maximum-likelihood estimator, labels those regions according to fit quality, and iteratively retrains the ANN.

---

## 5. Canonical Published ANN

The faithful baseline should begin with the architecture most directly supported by the paper:

```text
input window:        7 × 7 pixels
input features:      49 pixel values
hidden layers:       1
hidden units:        30
hidden activation:   sigmoid
output units:        1
output activation:   sigmoid
decision threshold:  0.5
spatial selection:   local maximum in a 5 × 5 neighborhood
```

The original conceptual forward pass is:

\[
z_m = \sum_i \Theta_{mi} P_i
\]

\[
a_m = \sigma(z_m)
\]

\[
z_{\mathrm{out}} = \sum_j \Theta_j a_j
\]

\[
a_{\mathrm{out}} = \sigma(z_{\mathrm{out}})
\]

with bias terms included.

The paper reports that increasing the hidden layer beyond roughly 30 nodes produced little additional improvement in the tested problem.

### Important historical inconsistency

Legacy CUDA snapshots are not perfectly aligned with the paper.

Examples include variants using:

- 7×7 input windows with 100 hidden units.
- 9×9 input windows with 30 hidden units.

Treat these as historical variants, not as silent replacements for the canonical paper reproduction.

The canonical reproduction should start with the published 7×7 / 30-hidden-unit model unless stronger evidence is documented.

---

## 6. Learning Loop

The canonical learning loop is iterative and dataset-adaptive.

At a high level:

```python
initialize detector randomly

while not converged:
    choose dataset / frames
    preprocess frames
    propose candidates with detector
    localize candidates
    label candidates from fit quality
    add examples to training memory
    retrain detector
```

The paper describes safeguards against a pathological first iteration where a random detector produces only bad examples. If too few acceptable fits are found early, the original method can reinitialize and try again until enough positive examples exist to begin learning.

Preserve this concept in the faithful reproduction, even if the modern implementation exposes a configurable alternative.

---

## 7. Training Memory / Replay

The historical code accumulated examples across iterations and discarded a small fraction to prevent memory growth.

The modern implementation should make this explicit.

Use a bounded replay/training buffer with provenance.

Each stored example should be able to retain at least:

```text
source dataset / frame
candidate x/y coordinate
detector score
image patch or retrievable source reference
fit parameters
fit uncertainty / CRLB
log likelihood
final good/bad label
training iteration
```

Support the historical behavior as closely as practical, but implement cleaner policies separately.

Potential modern extensions may include:

- bounded positive/negative buffers,
- hard-negative retention,
- class balancing,
- reservoir sampling,
- configurable eviction strategies.

Do not silently replace the historical strategy in the canonical reproduction.

---

## 8. Localization Model

The canonical localizer is based on a Gaussian approximation to a molecular point-spread function and a maximum-likelihood fit.

The historical implementation estimates parameters including:

```text
x position
y position
photon count / amplitude-like photon parameter N
sigma_x
sigma_y
background / offset
```

The pixel model should preserve the use of a **pixel-integrated Gaussian**, rather than replacing it with a point-sampled Gaussian without documentation.

A representative separable pixel-integrated model is:

\[
\mu_{ij}
=
N E_x(i; x,\sigma_x)
  E_y(j; y,\sigma_y)
+
b
\]

where each \(E\) term represents the Gaussian integral across the finite pixel boundary.

The implementation should support Poisson-likelihood-based fitting, Fisher-information calculation, and CRLB-style parameter uncertainty estimates consistent with the historical method.

### MLE implementation principle

Start with a readable, mathematically explicit reference implementation.

Do not optimize away clarity before correctness tests exist.

---

## 9. Fit-Quality Oracle

MINuTIA depends on the localizer producing a meaningful good/bad decision.

The historical code uses configurable tolerances involving fitted parameters and reported uncertainties.

Examples of historical criteria include bounds or fractional uncertainty on:

- photon count,
- background,
- x/y localization uncertainty,
- sigma values,
- sigma uncertainty,
- fit position relative to the candidate center.

Do not hard-code a single unexplained “quality score.”

Represent fit quality as a configurable, inspectable policy.

The API should make the oracle concept explicit, for example:

```python
labels = quality_oracle(fits, uncertainties, config)
```

The faithful reproduction should preserve a tolerance-based mode.

Experimental quality models may be added separately.

---

## 10. Central Modernization Goal: Eliminate the Old Detector→Localizer Round Trip

The historical implementation performs work roughly like this:

```text
GPU ANN inference
        ↓
full activation image
        ↓
host-side peak selection
        ↓
host-side candidate loops
        ↓
crop candidate ROI
        ↓
assemble ROI tensor
        ↓
send ROI tensor to localization GPU code
        ↓
MLE
```

This introduces avoidable work:

- accelerator→host synchronization,
- full activation-map materialization,
- host-side candidate traversal,
- explicit re-segmentation,
- ROI copies,
- another accelerator submission.

The modern fast path should instead be designed as:

```text
                     DEVICE MEMORY

raw / preprocessed frame batch
        ↓
detector
        ↓
threshold + local-max suppression
        ↓
candidate coordinate compaction
        ↓
(frame, x, y, score) candidates
        ↓
batched MLE reads source frame directly
        ↓
fit parameters + CRLB + log likelihood
        ↓
fit-quality labels
        ↓
training / replay update
```

The important abstraction is:

```python
fits = localize(
    frames=device_frames,
    candidates=device_candidates,
)
```

not:

```python
rois = crop_candidates(frames, candidates)
fits = localize(rois)
```

The localizer already knows its support window. In the preferred architecture, it should read the relevant pixels directly from the original device-resident frame tensor.

### ROI gathering is an optimization, not an architectural boundary

Profiling may show that materializing a compact contiguous ROI tensor improves memory access enough to be faster.

That is allowed.

But:

- it must be justified by benchmark data,
- it should remain on-device,
- it should not become a mandatory CPU-visible intermediate,
- it should not obscure the logical detector→coordinate→localizer flow.

---

## 11. Portable Reference First, Custom Kernels Later

Do not begin by writing custom CUDA.

The first correct implementation should use normal PyTorch tensors and operations.

A reasonable first accelerator path is:

```text
PyTorch tensor on device
        ↓
preprocessing
        ↓
MLP detector
        ↓
max_pool2d / equivalent NMS
        ↓
coordinate extraction / compaction
        ↓
vectorized or batched MLE
        ↓
vectorized CRLB / quality evaluation
```

The goal is to preserve device residency before introducing native kernels.

This should work, where practical, on:

- CPU,
- CUDA,
- ROCm-compatible PyTorch environments.

Only after reference correctness and profiling exist should the project consider:

- Triton,
- C++/CUDA extensions,
- fused detector/NMS kernels,
- fused candidate→MLE kernels,
- specialized Fisher/CRLB kernels.

Any native kernel must have a reference implementation and numerical equivalence tests.

---

## 12. Repository Layout

Prefer this general structure:

```text
.
├── AGENTS.md
├── LICENSE
├── NOTICE
├── README.md
├── CITATION.cff
├── pyproject.toml
├── docs/
│   ├── original_algorithm.md
│   ├── architecture.md
│   ├── paper_transcription.md
│   ├── paper_notes.md
│   └── roadmap.md
├── legacy/
│   ├── README.md
│   └── matlab/
├── src/
│   └── minutia/
│       ├── sim/
│       ├── preprocess/
│       ├── detector/
│       ├── localization/
│       ├── quality/
│       ├── training/
│       └── pipeline/
├── tests/
├── benchmarks/
├── experiments/
└── scripts/
```

Avoid unnecessary framework layers.

Modules should correspond to real scientific or engineering boundaries.

---

## 13. Recommended Python Stack

Primary implementation:

```text
Python 3.12+
PyTorch
NumPy
SciPy
```

Use additional dependencies only when they materially improve the project.

Expected tooling:

```text
uv
pyproject.toml
pytest
ruff
mypy or pyright if useful
GitHub Actions
```

Use type hints for public APIs.

Keep notebooks optional and explanatory. Core behavior must live in importable/testable Python modules, not notebooks.

---

## 14. Synthetic Data Is a First-Class Requirement

The repository must work without private laboratory datasets.

Provide synthetic data generation capable of producing known ground truth.

At minimum support:

- ordinary symmetric Gaussian-like PSFs,
- variable photon counts,
- variable backgrounds,
- sub-pixel x/y offsets,
- noise consistent with the modeled acquisition process,
- astigmatic PSFs with independently varying x/y widths.

The simulator should expose the true parameters so detector/localizer correctness can be quantitatively tested.

Synthetic data is not merely demo material; it is part of the validation infrastructure.

---

## 15. Canonical Experiments

The project should eventually support reproducible experiments for:

### 15.1 Detector learning

Measure detector behavior over iterative MINuTIA training.

Track:

- detection efficiency / recall against known simulated molecules,
- false-positive rate,
- positive-fit fraction,
- number of generated examples,
- loss / detector calibration where useful.

### 15.2 SNR / photon-count regimes

Measure behavior as photon count and background vary.

The original paper reports that missed molecules tended to have lower photon counts and poorer signal conditions than successfully detected molecules. The modern project should reproduce this kind of analysis where possible.

### 15.3 Astigmatic PSF adaptation

Show that the same general MINuTIA loop can adapt to altered PSF shapes without requiring a separately hand-labeled detector training set.

### 15.4 Localization accuracy

Measure fitted parameter error against known synthetic truth:

```text
x error
y error
N error
sigma_x error
sigma_y error
background error
```

### 15.5 Performance

Measure at least:

```text
CPU reference throughput
accelerator PyTorch throughput
staged detector→ROI→MLE throughput
device-resident detector→coordinate→MLE throughput
```

Do not claim performance improvements until measurements support them.

---

## 16. Tests Required Before Optimization

At minimum, establish tests for:

### Simulation

- deterministic seeded generation,
- known center placement,
- expected qualitative PSF behavior,
- astigmatic width variation.

### Pixel-integrated Gaussian

- symmetry,
- translation behavior,
- numerical sanity,
- comparison against numerical integration or trusted reference cases.

### Detector

- exact forward-pass agreement with a simple reference,
- bias handling,
- 7×7 shape assumptions,
- threshold behavior,
- local-maximum suppression.

### MLE

- recovers known x/y within expected tolerance,
- recovers known photon/background parameters,
- behaves sensibly across SNR regimes,
- does not silently produce invalid parameter values.

### Fisher / CRLB

- finite results on valid fits,
- symmetric Fisher matrix,
- covariance / uncertainty behavior improves with stronger signal where expected,
- failure states are explicit.

### Quality oracle

- deterministic pass/fail behavior,
- boundary conditions,
- correct handling of invalid fits.

### End-to-end

- one complete MINuTIA iteration,
- replay-buffer update,
- detector parameters change after training,
- deterministic behavior under fixed seed where practical.

### Device equivalence

Where GPU paths exist:

- CPU and accelerator outputs should agree within documented numerical tolerances.

---

## 17. Historical Material

Legacy code should be curated, not dumped blindly.

Useful historical source categories include:

```text
Neural_Learning.m
func_Quhzx_*_for_learning.m
app_gpu_tol_all_color_learning.m
func_neural_teach.m
neuralcostfunc.m
neural_gpu.m
image_neural_*.cu
func_mle_crlb.m
slim_locs.m
slim_chain_loc_*.cu
divide_up.m
```

`legacy/README.md` should map each included file to its historical role.

For each file, document:

```text
source path
approximate date / generation if known
role in original pipeline
whether it appears consistent with the publication
known divergences
license / provenance status
whether modern code depends on it
```

Modern production code must not import or execute MATLAB.

---

## 18. Provenance and Licensing

The modern implementation is intended to be licensed under Apache-2.0.

Do **not** assume that every historical MATLAB/CUDA file is automatically covered by the modern repository license.

Before adding legacy code to the public repository:

1. Identify authorship where possible.
2. Preserve existing copyright/license notices.
3. Do not relicense third-party code.
4. If provenance is uncertain, prefer documenting the historical behavior and independently reimplementing it in `src/`.
5. Keep uncertain third-party code out of the public repository when necessary.

Do not bundle the publisher PDF merely because the repository is open source.

The paper should be cited by bibliographic reference and DOI.

---

## 19. Faithful Reproduction vs Modern Extensions

Keep a hard conceptual boundary between:

### Canonical reproduction

The behavior needed to reproduce the published method:

```text
7×7 local region
single hidden-layer ANN
30 hidden sigmoid units
sigmoid output
0.5 threshold
5×5 local-max criterion
Gaussian MLE localizer
fit-quality-derived labels
iterative retraining
```

### Engineering modernization

Allowed without changing scientific intent:

```text
clean Python package
PyTorch
device residency
vectorization
batched MLE
bounded replay buffers
CI
tests
configuration files
reproducible experiments
profiling
clean GPU memory flow
```

### Experimental extensions

Must be labeled clearly:

```text
CNN detector
deeper MLP
learned quality oracle
multi-emitter fitting
different PSF models
alternative background estimators
different replay strategies
fully fused custom GPU kernels
end-to-end differentiable variants
```

Experimental improvements should not overwrite the canonical baseline.

---

## 20. Do Not Misrepresent the Published Model

The published model is a single-hidden-layer feed-forward ANN operating on local pixel values.

Do not casually describe the original implementation as a convolutional neural network merely because the network is scanned spatially across an image.

A modern CNN extension is welcome, but it must be called an extension.

Likewise, prefer precise terms such as:

- iterative self-training,
- fit-guided supervision,
- localization-guided pseudo-labeling,
- physics/model-guided training,

when describing the modern interpretation.

When discussing the historical paper, preserve its original framing where relevant.

---

## 21. Performance Engineering Principles

When optimizing:

1. Profile first.
2. Measure device synchronization.
3. Measure memory traffic.
4. Avoid host-visible intermediates.
5. Avoid full-frame intermediate tensors if coordinates or sparse representations suffice.
6. Preserve a readable reference path.
7. Add benchmarks before and after each major optimization.
8. Never trade correctness for unmeasured speed.

Potential hot paths include:

```text
background preprocessing
sliding detector inference
NMS / candidate compaction
candidate pixel access
MLE iterations
Fisher / CRLB evaluation
training-example extraction
```

The highest-value optimization target is expected to be the boundary between detection and localization.

---

## 22. Candidate Representation

Use an explicit candidate representation.

At minimum:

```text
frame_index
x
y
detector_score
```

Potentially:

```text
batch_index
channel
scale / window metadata
```

Candidates should remain on-device in the fast path.

Avoid Python object lists inside performance-critical GPU loops.

---

## 23. Failure Handling

Scientific code should make failure states explicit.

Examples:

```text
non-converged fit
NaN parameters
non-positive photon estimate
singular Fisher matrix
candidate outside valid image support
invalid sigma
invalid background
insufficient positive examples
empty candidate set
```

Do not silently coerce all failures into negative training examples without documenting that policy.

The label oracle should distinguish:

```text
valid negative
invalid fit / numerical failure
valid positive
```

even if the canonical training loop later maps some of those states together.

---

## 24. Reproducibility

All experiments should support deterministic or documented pseudo-deterministic execution where practical.

Record:

```text
random seed
software version / git commit
device
dtype
dataset-generation configuration
model configuration
fit configuration
quality tolerances
training iteration settings
```

Experiment outputs should be machine-readable where practical.

Do not rely on screenshots or notebook state as the only record of a result.

---

## 25. Numerical Precision

Do not assume `float32` is always sufficient for localization and Fisher calculations.

Establish a precision policy empirically.

A good initial approach:

- simulator: configurable float32/float64,
- detector: float32 by default,
- reference MLE / Fisher: float64 initially,
- optimized accelerator MLE: benchmark float32 and float64 behavior,
- mixed precision only after numerical validation.

Document tolerances used in CPU/GPU equivalence tests.

---

## 26. Background Processing

The publication used rolling-ball subtraction.

The faithful baseline should include a rolling-ball-equivalent preprocessing mode or another implementation clearly documented as reproducing its effect.

Alternative preprocessing methods are allowed as experiments.

Do not let background processing become an undocumented hidden dependency of detector performance.

---

## 27. Convergence

The original paper describes continuing the iterative process until the ANN identifies successfully fitted regions at a user-defined rate or the process is manually terminated.

The modern implementation should expose explicit convergence criteria.

Examples:

```text
target successful-fit fraction
maximum iterations
minimum improvement over N iterations
maximum training examples
maximum runtime
```

Canonical experiments should report which criterion was used.

---

## 28. Documentation Deliverables

Maintain the following documents:

### `docs/original_algorithm.md`

A clean reconstruction of the original MINuTIA algorithm from the publication and legacy source.

### `docs/architecture.md`

Show:

1. historical pipeline,
2. faithful modern reference architecture,
3. device-resident optimized architecture.

### `docs/paper_transcription.md`

A clean transcription/reference representation of the paper.

### `docs/paper_notes.md`

Modern commentary, historical inconsistencies, interpretation notes, and implementation decisions.

### `docs/roadmap.md`

Concrete implementation milestones and current status.

Do not use `AGENTS.md` as a changelog.

---

## 29. README Expectations

The README should quickly explain the project to both:

- a computer-vision / ML engineer,
- a scientific-imaging researcher.

The opening visual should eventually communicate:

```text
raw frame
  ↓
detector response
  ↓
candidate
  ↓
MLE fit
  ↓
accepted/rejected training label
  ↓
improved detector
```

The README should clearly distinguish:

- the 2018 method,
- the modern reimplementation,
- the optimized device-resident path,
- experimental extensions.

Include the publication citation prominently.

---

## 30. Development Sequence

Unless there is a strong reason to deviate, implement in roughly this order:

### Phase 0 — Archaeology

- curate legacy files,
- document provenance,
- reconstruct original data flow,
- document implementation drift.

### Phase 1 — Scientific reference

- synthetic PSF simulator,
- pixel-integrated Gaussian,
- Poisson likelihood,
- reference MLE,
- Fisher matrix / CRLB,
- fit-quality oracle.

### Phase 2 — Detector reference

- canonical 7×7 / 30-unit ANN,
- detector training,
- thresholding,
- 5×5 NMS,
- tests against simple reference calculations.

### Phase 3 — MINuTIA loop

- randomized frame selection,
- candidate localization,
- label generation,
- replay/training buffer,
- iterative retraining,
- convergence logic.

### Phase 4 — Reproduction experiments

- low/mid/high signal regimes,
- astigmatic PSFs,
- learning curves,
- detection/false-positive metrics.

### Phase 5 — Device-resident pipeline

- keep frames on accelerator,
- on-device NMS,
- on-device candidate representation,
- localizer reads source frames directly,
- avoid host round trips,
- benchmark against staged pipeline.

### Phase 6 — Optimization

Only after profiling:

- vectorization improvements,
- `torch.compile` where useful,
- Triton or native extensions,
- kernel fusion.

### Phase 7 — Extensions

- CNN detector,
- alternative PSF/localization models,
- multi-emitter support,
- more advanced replay policies.

---

## 31. Agent Working Rules

When operating autonomously:

1. **Read the relevant paper/legacy documentation before changing scientific behavior.**
2. **Do not rewrite legacy files as part of cleanup.** Treat them as evidence.
3. **Add tests with every mathematical component.**
4. **Prefer small coherent commits.**
5. **Document ambiguity instead of hiding it.**
6. **Do not make performance claims without benchmarks.**
7. **Do not make scientific accuracy claims without experiments.**
8. **Keep canonical reproduction paths usable while adding optimized paths.**
9. **Avoid adding dependencies merely for convenience.**
10. **Do not conflate modernization with methodological changes.**
11. **If an implementation decision changes the scientific algorithm, label it experimental and explain why.**
12. **Keep the public repository reproducible without private data.**

---

## 32. Definition of “Done” for a Feature

A scientific/algorithmic feature is not complete until it has:

```text
implementation
tests
documentation
configuration where appropriate
failure handling
reproducible example or experiment where appropriate
```

A performance feature additionally requires:

```text
baseline benchmark
optimized benchmark
numerical-equivalence validation
hardware/software environment record
```

---

## 33. Project Identity

The strongest technical story in this repository is not:

> “an old neural network was rewritten with a newer neural network.”

The important idea is:

> **A learned proposal mechanism is trained by the success or failure of a downstream physical/statistical estimator, and the entire proposal→estimation→supervision loop can be made accelerator-resident.**

Preserve that identity throughout the architecture and documentation.

---

## 34. Canonical Citation

Use:

```text
Nelson, A. J., & Hess, S. T. (2018).
Molecular imaging with neural training of identification algorithm
(neural network localization identification).
Microscopy Research and Technique.
https://doi.org/10.1002/jemt.23059
```

Also maintain a `CITATION.cff` file for the repository.

---

## 35. Final Principle

**Reproduce first. Measure second. Optimize third. Extend fourth.**

The modern repository should make it possible to understand:

1. what MINuTIA originally did,
2. why it worked,
3. how closely the modern implementation reproduces it,
4. where the historical implementation was inefficient,
5. how a modern device-resident architecture improves the systems design,
6. which later ideas are faithful modernization versus new research.
