# Reproducibility notes

## Recorded environment

The latest notebook was executed under WSL using Python 3.12.3 and CUDA on an NVIDIA RTX 4060 Laptop GPU. Direct package versions are listed in requirements.txt. GPU driver, CUDA wheel availability, and deterministic kernel behavior can vary across systems.

The default configuration uses seed 42. Full deterministic mode is optional because it can reduce performance and may not cover every CUDA operation.

## EdgeFace dependency

The hybrid preset requires the official EdgeFace source and the edgeface_s_gamma_05 checkpoint. The source is pinned through third_party/edgeface. Initialize it with:

~~~bash
git submodule update --init --recursive
~~~

The core configuration points to third_party/edgeface by default. If a different checkout is required, override edgeface_dir in the experiment configuration.

## Evaluation protocol implemented in code

- Four pair-verification benchmarks: LFW, AgeDB-30, CALFW, and CPLFW.
- Euclidean distance between normalized embeddings.
- Ten-fold cross-validation.
- Threshold selected on each training-fold partition and applied to its held-out fold.
- Mean accuracy, fold standard deviation, and mean selected threshold reported.

## Verification boundary

The committed notebook retains outputs from the development machine, but no dataset or checkpoint is included. Therefore a clean clone can validate repository structure and source syntax immediately, while numerical reproduction requires independently obtained datasets and a complete training run.
