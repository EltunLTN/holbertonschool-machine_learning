#!/usr/bin/env python3
"""Defines a function that calculates the Q affinities for t-SNE."""
import numpy as np


def Q_affinities(Y):
    """Calculates the Q affinities.

    Args:
        Y (numpy.ndarray): array of shape (n, ndim) containing the
            low dimensional transformation of X, where n is the
            number of points and ndim is the new dimensional
            representation of X.

    Returns:
        tuple: (Q, num)
            Q (numpy.ndarray): array of shape (n, n) containing the
                Q affinities.
            num (numpy.ndarray): array of shape (n, n) containing
                the numerator of the Q affinities.
    """
    n, ndim = Y.shape
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.matmul(Y, Y.T), sum_Y).T, sum_Y)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
