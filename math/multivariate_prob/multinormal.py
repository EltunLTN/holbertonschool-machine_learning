#!/usr/bin/env python3
"""Multivariate Normal Distribution"""

import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initializes the MultiNormal distribution.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean: shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Covariance: shape (d, d)
        diff = data - self.mean
        self.cov = (diff @ diff.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) - the data point

        Returns:
            The value of the PDF at x
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Multivariate Normal PDF formula:
        # f(x) = 1 / sqrt((2π)^d * |Σ|) * exp(-0.5 * (x-μ)^T Σ^{-1} (x-μ))

        # Determinant and inverse of covariance matrix
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Normalization coefficient
        coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        # Exponent: -0.5 * (x - μ)^T Σ^{-1} (x - μ)
        diff = x - self.mean
        exponent = -0.5 * (diff.T @ inv @ diff).item()

        return coefficient * np.exp(exponent)
