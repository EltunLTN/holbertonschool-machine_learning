#!/usr/bin/env python3
"""Correlation Matrix"""

import numpy as np


def correlation(C):
    """Calculates a correlation matrix from a covariance matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Standard deviations: sqrt of diagonal (variances)
    std = np.sqrt(np.diag(C))

    # Outer product gives σ_i * σ_j for each (i, j)
    outer = np.outer(std, std)

    # ρ_ij = σ_ij / (σ_i * σ_j)
    return C / outer
