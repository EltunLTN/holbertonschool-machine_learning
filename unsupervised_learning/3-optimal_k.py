#!/usr/bin/env python3
"""Module for determining the optimal number of clusters for K-Means."""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """Evaluate K-Means clustering quality using inertia and silhouette scores.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        max_clusters (int): Maximum number of clusters to evaluate (>=2).
        random_state (int): Random seed for reproducibility.

    Returns:
        list[int]: Evaluated cluster numbers from 2 to max_clusters.
        list[float]: Inertia values for each cluster number (elbow method).
        list[float]: Silhouette scores for each cluster number.
    """
    ks = list(range(2, max_clusters + 1))
    inertia_values = []
    silhouette_values = []

    for k in ks:
        model = K_Means(X, n_clusters=k, random_state=random_state)
        inertia_values.append(model.inertia_)
        score = metrics.silhouette_score(X, model.labels_)
        silhouette_values.append(score)

    return ks, inertia_values, silhouette_values
