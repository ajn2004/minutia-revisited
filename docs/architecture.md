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

The reference uses ordinary PyTorch operations and works on CPU.

## Device-resident design

```text
device frames → detector → device threshold/NMS/compaction
              → (frame, x, y, score) candidates
              → direct source-frame reads → batched MLE/CRLB
              → device labels and training tensors
```

No CPU-visible heatmap or mandatory ROI tensor is required. A compact device
gather may be added after profiling, but it is an optimization rather than a
conceptual stage. Custom kernels wait for numerical equivalence and benchmark
evidence.
