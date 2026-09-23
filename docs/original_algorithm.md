# Original MINuTIA algorithm

MINuTIA (Nelson & Hess, 2018) trains an image-region detector using the
success or failure of a downstream localization fit as its supervision. It is
fit-guided iterative self-training, not an ordinary manually labelled object
classifier.

1. Read frames and subtract background (the paper specifies rolling-ball subtraction).
2. Evaluate a feed-forward ANN independently around each central pixel.
3. Use a 7×7 window (49 values), one sigmoid hidden layer of 30 units, and a
   sigmoid scalar output, including biases.
4. Retain responses at least 0.5 that are local maxima in a 5×5 neighbourhood.
5. Fit proposed regions with a pixel-integrated Gaussian Poisson MLE.
6. Calculate log likelihood and Fisher-information/CRLB uncertainties.
7. Apply inspectable tolerances to fit parameters, uncertainties, and fit position.
8. Store positive/negative examples and retrain the ANN.
9. Repeat over randomized datasets and frames until configured convergence.

A random first detector can produce no good fits. The modern API therefore
supports a finite minimum-positive-example retry policy and detector
reinitialization. Its bounded `ReplayBuffer` is a modern uniform-retention
policy, not a reconstruction claim about historical example discarding. The
current portable preprocessing is explicitly named
`rolling_ball_approximation`; it targets the paper's rolling-ball step but uses
a local mean until a calibrated implementation is available.
Historical variants are documented in `legacy/README.md` and `docs/paper_notes.md`.
