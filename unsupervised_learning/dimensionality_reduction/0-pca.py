#!/usr/bin/env python3
"""Defines a function that performs PCA on a dataset to maintain
a given fraction of its original variance."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): array of shape (n, d) where n is the
            number of data points and d is the number of dimensions
            in each point. All dimensions have a mean of 0 across
            all data points.
        var (float): fraction of the variance that the PCA
            transformation should maintain.

    Returns:
        numpy.ndarray: the weights matrix W of shape (d, nd), where
            nd is the new dimensionality of the transformed X, that
            maintains var fraction of X's original variance.
    """
    u, s, vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    nd = np.argwhere(cum_var >= var)[0, 0] + 1
    W = vh[:nd].T
    return W
