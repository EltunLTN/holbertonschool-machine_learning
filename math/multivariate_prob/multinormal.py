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
