#!/usr/bin/env python3
"""Performs K-means on a dataset"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
        iterations that should be performed

    If no change in the cluster centroids occurs between iterations, the
        function returns early.
    If a cluster contains no data points during the update step, its
        centroid is reinitialized.

    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)

    C = np.random.uniform(low, high, size=(k, d))

    for i in range(iterations):
        C_prev = C.copy()

        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Recompute centroids
        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = np.random.uniform(low, high, size=(1, d))
            else:
                C[j] = X[clss == j].mean(axis=0)

        if np.array_equal(C, C_prev):
            break

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
