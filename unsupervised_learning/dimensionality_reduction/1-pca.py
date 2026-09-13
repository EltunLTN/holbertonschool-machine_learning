#!/usr/bin/env python3
"""Defines a function that performs PCA on a dataset to reduce it
to a specified new dimensionality."""
import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset to reduce it to ndim dimensions.

    Args:
        X (numpy.ndarray): array of shape (n, d) where n is the
            number of data points and d is the number of dimensions
            in each point.
        ndim (int): the new dimensionality of the transformed X.

    Returns:
        numpy.ndarray: T, array of shape (n, ndim) containing the
            transformed version of X.
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T
    T = np.matmul(X_m, W)
    return T
