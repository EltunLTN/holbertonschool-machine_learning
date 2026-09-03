#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    X is a numpy.ndarray of shape (n, d) containing the data points whose
        PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the
        distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
        distribution

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for
            each data point
        All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0]:
        return None
    if S.shape[0] != S.shape[1] or S.shape[0] != m.shape[0]:
        return None

    n, d = X.shape

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    diff = X - m
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

    denom = np.sqrt(((2 * np.pi) ** d) * det)

    P = (1 / denom) * np.exp(exponent)
    P = np.maximum(P, 1e-300)

    return P
