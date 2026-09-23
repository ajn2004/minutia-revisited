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
