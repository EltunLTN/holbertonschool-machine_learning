#!/usr/bin/env python3
"""Defines a function that calculates the gradients of Y for t-SNE."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): array of shape (n, ndim) containing the
            low dimensional transformation of X.
        P (numpy.ndarray): array of shape (n, n) containing the P
            affinities of X.

    Returns:
        tuple: (dY, Q)
            dY (numpy.ndarray): array of shape (n, ndim) containing
                the gradients of Y.
            Q (numpy.ndarray): array of shape (n, n) containing the
                Q affinities of Y.
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    PQ_diff = P - Q
    dY = np.zeros((n, ndim))
    for i in range(n):
        dY[i] = np.sum(
            (PQ_diff[:, i] * num[:, i])[:, np.newaxis] * (Y[i] - Y),
            axis=0
        )
    return dY, Q
