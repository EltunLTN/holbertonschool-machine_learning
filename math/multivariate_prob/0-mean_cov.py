#!/usr/bin/env python3
"""Mean and Covariance"""

import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set

    Returns:
        mean: numpy.ndarray of shape (1, d) - mean of the data set
        cov:  numpy.ndarray of shape (d, d) - covariance matrix
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Mean: shape (1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Covariance: (X - mean)^T @ (X - mean) / (n - 1)
    diff = X - mean
    cov = (diff.T @ diff) / (n - 1)

    return mean, cov
