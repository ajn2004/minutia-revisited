# Paper notes and implementation decisions

The paper describes a 7×7 input and 30 hidden nodes. `Neural_Learning.m`
matches this with 30×50 first-layer parameters and 1×31 output parameters,
where the extra dimension is the bias. CUDA snapshots disagree: one uses 100
hidden nodes and another loops over 9×9 inputs while retaining 30 nodes. The
modern baseline follows the paper and matching MATLAB learner.

The paper says responses above a 0.5 sigmoid cutoff are selected, while one
legacy comment describes a cutoff of zero. The implementation follows 0.5.

Historical code uses rolling-ball subtraction, hand-written Newton-like
updates, and explicit Fisher inverse expressions. The modern reference uses
the same pixel-integrated model with readable tensor linear algebra and
explicit validity flags; it is not a claim of bitwise identity.
