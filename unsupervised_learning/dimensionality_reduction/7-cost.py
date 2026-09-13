#!/usr/bin/env python3
"""Defines a function that calculates the cost of the t-SNE
transformation."""
import numpy as np


def cost(P, Q):
    """Calculates the cost of the t-SNE transformation.

    Args:
        P (numpy.ndarray): array of shape (n, n) containing the P
            affinities.
        Q (numpy.ndarray): array of shape (n, n) containing the Q
            affinities.

    Returns:
        float: C, the cost of the transformation.
    """
    Q = np.where(Q < 1e-12, 1e-12, Q)
    P = np.where(P < 1e-12, 1e-12, P)
    C = np.sum(P * np.log(P / Q))
    return C
