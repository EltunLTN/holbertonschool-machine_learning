# Dimensionality Reduction

This project implements two dimensionality reduction techniques from
scratch using only `numpy`: **PCA** (Principal Component Analysis)
and **t-SNE** (t-Distributed Stochastic Neighbor Embedding).

## Requirements

* Ubuntu 20.04 LTS, Python 3.9
* `numpy` 1.25.2
* Style: `pycodestyle` 2.11.1
* Every file starts with `#!/usr/bin/env python3`, ends with a
  newline, is executable, and is fully documented (module, class,
  and function docstrings).
* No imports besides `numpy` are used, and functions imported
  between files are explicitly noted per task.

## Files

| File                  | Description |
|-----------------------|-------------|
| `0-pca.py`            | `pca(X, var=0.95)` — PCA that keeps a given fraction of the variance, returning the weights matrix `W`. |
| `1-pca.py`            | `pca(X, ndim)` — PCA that reduces `X` to a fixed number of dimensions, returning the transformed data `T`. |
| `2-P_init.py`         | `P_init(X, perplexity)` — initializes the pairwise squared distance matrix `D`, the affinity matrix `P`, the `beta` values, and the target Shannon entropy `H`. |
| `3-entropy.py`        | `HP(Di, beta)` — computes the Shannon entropy and P affinities for a single point's distance row. |
| `4-P_affinities.py`   | `P_affinities(X, tol=1e-5, perplexity=30.0)` — binary-searches `beta` per point to match the target perplexity, then symmetrizes and normalizes `P`. |
| `5-Q_affinities.py`   | `Q_affinities(Y)` — computes the low-dimensional Q affinities using the Student t-distribution kernel. |
| `6-grads.py`          | `grads(Y, P)` — computes the gradient of the t-SNE cost with respect to `Y`. |
| `7-cost.py`           | `cost(P, Q)` — computes the KL-divergence cost between `P` and `Q`. |
| `8-tsne.py`           | `tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500)` — full t-SNE pipeline: PCA preprocessing, early exaggeration, and momentum-based gradient descent. |

## Data

Test scripts expect `mnist2500_X.txt` and `mnist2500_labels.txt` in
the working directory (a 2500-sample, 784-feature subset of MNIST).

## Usage

```bash
./0-main.py
./1-main.py
# etc.
```

Each `N-main.py` script is a standalone example that imports and
runs the corresponding `N-*.py` module.

## Notes on the algorithm

* **PCA** uses SVD (`np.linalg.svd`) on the mean-centered data; the
  right singular vectors give the principal directions, and the
  cumulative explained-variance ratio picks how many components to
  keep.
* **t-SNE** follows Algorithm 1 of van der Maaten & Hinton (2008):
  * P affinities are computed per-point with a binary search on
    `beta = 1 / (2 * sigma^2)` so every conditional distribution
    matches the target perplexity within `tol`.
  * Q affinities use the Student t-distribution (1 degree of
    freedom) as a heavier-tailed kernel in the low-dimensional
    space.
  * The gradient descent uses momentum (`0.5` for the first 20
    iterations, `0.8` after) and early exaggeration (P values
    multiplied by 4 for the first 100 iterations) to help escape
    poor local optima early in training.
  * `Y` is re-centered (mean subtracted) after every iteration.
