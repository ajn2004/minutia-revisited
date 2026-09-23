# Historical source map

The files in this directory are archaeological research code. They are not
imported or executed by the modern Python implementation. Provenance and
licensing of individual snapshots should be reviewed before redistribution;
the Apache-2.0 license for the modern code does not automatically relicense
these files.

| File | Historical role | Notes |
| --- | --- | --- |
| `Neural_Learning.m` | top-level iterative learner | Clearest 7×7/30 canonical variant; uses 30×50 and 1×31 parameter arrays. |
| `func_neural_teach.m`, `neuralcostfunc.m` | ANN optimization and regularized cost | MATLAB training helpers. |
| `neural_gpu.m` | MATLAB wrapper for detector MEX | Host/device boundary. |
| `image_neural_3.cu` | detector MEX snapshot | Active calculation has 30 hidden units; comments retain stale 100-unit text. |
| `image_neural_norm.cu` | detector MEX snapshot | 7×7 detector with 100 hidden units; historical drift. |
| `func_Quhzx_02_7_for_learning.m` | preprocessing, detection, segmentation, localization | Rolling-ball and explicit host ROI construction. |
| `func_mle_crlb.m` | readable MATLAB MLE/CRLB prototype | Pixel-integrated Gaussian and six-parameter Fisher calculation. |
| `slim_chain_loc_13.cu` | CUDA batched localizer | Receives a prepared ROI stack. |
| `slim_locs.m` | localization wrapper | Filters failed fits and manages batches. |
| `app_gpu_tol_all_color_learning.m` | fit-quality tolerance application | Parameter, uncertainty, and candidate-centre criteria. |
| `func_build_gauss.m` | synthetic Gaussian generation | Historical simulation helper. |
| `divide_up.m` | data partition helper | Operational utility. |

## Known divergences

The publication and `Neural_Learning.m` support the canonical 7×7 input,
single 30-unit sigmoid hidden layer, sigmoid output, 0.5 decision threshold,
and 5×5 local maximum. `image_neural_norm.cu` uses 100 hidden units, while a
different CUDA snapshot contains a 9×9 loop with 30 hidden units. These remain
variants, not silent replacements.

The old path materializes detector images, performs peak selection and
segmentation on the host, then sends copied ROIs to the localizer. The modern
pipeline exposes `(frame, x, y, score)` tensors and reads fitting windows from
source frames directly.
