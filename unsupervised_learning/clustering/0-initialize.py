#!/usr/bin/env python3
"""Initialize cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X is a numpy.ndarray of shape (n, d) containing the dataset that will
        be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)

    centroids = np.random.uniform(low, high, size=(k, d))
    return centroids
