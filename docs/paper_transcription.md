# Paper transcription reference

This document is a compact implementation reference for the published method;
it is not a reproduction of the publisher's PDF.

**Source:** A. J. Nelson and S. T. Hess, “Molecular imaging with neural
training of identification algorithm (neural network localization
identification),” *Microscopy Research and Technique* (2018).
[DOI: 10.1002/jemt.23059](https://doi.org/10.1002/jemt.23059)

## Transcribed algorithmic facts

- Rolling-ball background subtraction precedes detector evaluation.
- A feed-forward ANN evaluates a local 7×7 pixel region around each pixel.
- The canonical network has one sigmoid hidden layer with 30 units and one
  sigmoid output; the response threshold is 0.5.
- Responses are reduced to spatial local maxima in a 5×5 neighborhood.
- Candidate regions are localized with a Gaussian maximum-likelihood model.
- Fit quality supplies positive or negative training labels, after which the
  ANN is retrained iteratively.
- The process continues until the configured successful-fit criterion is met or
  the user stops it.

## Scope of this transcription

The implementation details are interpreted in `docs/original_algorithm.md` and
`docs/paper_notes.md`. Historical MATLAB and CUDA snapshots contain
architecture and preprocessing drift; those variants are not silently folded
into this canonical transcription.
