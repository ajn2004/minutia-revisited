# Nelson & Hess (2018) Figure 5 reproduction

This package runs seeded, synthetic, paper-guided simulations through the
canonical 7x7 / 30-hidden-unit sigmoid detector, Gaussian Poisson MLE,
tolerance oracle, and iterative fit-guided training loop.

```bash
python -m experiments.reproduce_2018.run \
  --config experiments/reproduce_2018/configs/smoke.toml --output out/smoke
```

The output contains `metadata.json`, `results.json`, `results.csv`, and a
four-panel `figure5_style.png` generated from the CSV. Full configurations
target 10,000 frames; `smoke.toml` is intentionally small for CI.

Each results row contains separate detector (pre-tolerance) and accepted
(post-tolerance) TP/FP/FN counts and rates. The Figure 5-style plot uses the
accepted rates; `fit_success_fraction` remains the independent fraction of
detector identifications passing the quality oracle. Rows also include seconds
for simulation, preprocessing/detection/NMS, localization, quality-oracle,
truth-matching/metrics, replay/training, and total iteration time.

Configurations default to the tensor-batched detector/localization path with
`execution_path = "batched"`. Set it to `"reference"` to use the readable
per-candidate localization oracle; the selected path is recorded in
`metadata.json`.

The reproduction's first iteration uses the historical bootstrap when
`bootstrap_enabled = true`: each attempt initializes every ANN parameter,
including biases, uniformly in `[-initialization_epsilon, +epsilon]`, analyzes
a new deterministic random subset of source frames, and accumulates fit-guided
examples without training. Training starts only after `bootstrap_min_positives`
positives exist;
the detector is reinitialized between attempts and an exhausted
`bootstrap_max_attempts` fails loudly. The historical learner's surviving
code uses a condition equivalent to more than 40 positives, but the numerical
threshold was not stated by the publication, so it is configurable. The smoke
configuration uses 3 solely to exercise this mechanism quickly.

## Terminology and formulas

Matching is frame-by-frame, one-to-one, and uses the configured Euclidean
radius. `TP`, `FP`, and `FN` are respectively matched identifications,
unmatched identifications, and unmatched known molecules. The rates are:

* identification precision = TP / (TP + FP)
* false-identification fraction = FP / (TP + FP)
* detection efficiency / recall = TP / known molecules
* fit-success fraction = tolerance-passing candidates / all candidates

The paper uses “detection accuracy”, “detection efficiency”, “false positive
rate”, and “successful fits” inconsistently; the CSV preserves explicit
formula names instead of silently equating them.

The primary `paper_methods` schedule uses 10 frames before iteration 15 and up
to 1,000 thereafter. `legacy_matlab` is also implemented: 10 frames for
iterations <5, 50 for 5-14, 500 for 15-24, then 1,000. The optional 5%
positive-example removal is retained as a historical replay policy.

Every iteration records selected source-frame indices (and bootstrap attempts
record theirs) in the JSON artifacts. Replay class counts and detector score
calibration diagnostics are included in each results row. Training uses
`training_mode = "modern_adam"` by default. The explicitly named
`historical_objective_lbfgs` mode uses a seeded 90% subset, BCE, weight-only L2
regularization with lambda 0.3, and at most 100 LBFGS iterations as a practical
approximation to the surviving MATLAB objective optimizer.
