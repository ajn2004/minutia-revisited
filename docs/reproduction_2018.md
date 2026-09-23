# Reproduction plan: Nelson & Hess (2018) Figure 5

`experiments/reproduce_2018` provides low, medium, and high signal-to-noise
regimes plus an astigmatic PSF. Every run is seeded and stores synthetic
molecule truth. The detector is the published 7x7, single-hidden-layer,
30-unit sigmoid ANN; localization uses the pixel-integrated Gaussian Poisson
MLE and configurable tolerance oracle.

## Explicitly stated in the paper

The paper describes simulation parameters sampled from distributions measured
from experimental FPALM data, iterative fit-guided detector learning, and the
four qualitative regimes represented here. It reports comparison targets of
approximately 170 photons for missed molecules and 230 for detected molecules,
and photon/sqrt(offset) averages near 69 and 73. These are comparison targets,
not tuning constraints.

The methods text says 10 frames initially and 1,000 after iteration 15; the
discussion says 100 to 1,000. The primary `paper_methods` schedule follows the
former. The historical alternative is named `legacy_matlab` rather than
silently reconciled.

## Recovered from historical code

`legacy/matlab/Neural_Learning.m` recovers the schedule 10 / 50 / 500 / 1,000
frames at iteration boundaries 5, 15, and 25, the 5% positive-example toss,
and tolerance values such as minimum photons 10, sigma 1-10, and fractional
uncertainty 0.5. `func_build_gauss.m` shows a fallback helper using background
0.5 and a sigma-related value 1.5, but it is not the Figure 5 FPALM
distribution.

## Modern assumptions

The publication does not provide numerical FPALM photon, background, and width
distributions, and the curated legacy sources do not recover them. Therefore
each config labels its surrogate distributions `paper_guided`. Their values
are explicit configuration, not claims about original data. The smoke case is
only an execution test. No configuration is tuned to force agreement with the
reported comparison targets.

Results include counts, rates, detected/missed photon summaries, the
photon/sqrt(background) proxy, seed, git commit, PyTorch version, and backend.
Plots are generated from `results.csv`, never hard-coded.
