#!/usr/bin/env python3
"""Module for standardizing tabular data using Scikit-learn preprocessing."""
from sklearn import preprocessing


def Standardize(X):
    """Standardize tabular data using Scikit-learn's StandardScaler.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized version of the input data,
            with the same shape as X.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
