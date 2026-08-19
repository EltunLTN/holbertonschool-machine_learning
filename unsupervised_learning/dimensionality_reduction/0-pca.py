#!/usr/bin/env python3
"""Module containing a function that performs PCA."""

import numpy as np


def pca(X, var=0.95):
    """Perform Principal Component Analysis on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        var: Fraction of variance to preserve.

    Returns:
        numpy.ndarray of shape (d, nd) containing the weights matrix.
    """
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    explained_variance = S ** 2
    explained_variance /= np.sum(explained_variance)

    cumulative_variance = np.cumsum(explained_variance)

    nd = np.searchsorted(cumulative_variance, var) + 1

    W = Vt[:nd].T

    return W
