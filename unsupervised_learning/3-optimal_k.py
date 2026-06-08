#!/usr/bin/env python3
"""Module for determining the optimal number of clusters."""

from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Determines the optimal number of clusters for K-Means.

    Args:
        X: numpy.ndarray of shape (n_samples, n_features)
        max_clusters: maximum number of clusters to test
        random_state: random seed

    Returns:
        ks: list of cluster numbers tested
        inertia_values: list of inertia values
        silhouette_values: list of silhouette scores
    """

    ks = []
    inertia_values = []
    silhouette_values = []

    for k in range(2, max_clusters + 1):
        kmeans = K_Means(
            X,
            n_clusters=k,
            random_state=random_state
        )

        ks.append(k)
        inertia_values.append(kmeans.inertia_)

        silhouette_values.append(
            metrics.silhouette_score(
                X,
                kmeans.labels_
            )
        )

    return ks, inertia_values, silhouette_values
