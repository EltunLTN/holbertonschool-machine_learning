#!/usr/bin/env python3
"""Module for Agglomerative Hierarchical Clustering using Scikit-learn."""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state, n_components,
                             use_pca_data=True):
    """Perform Agglomerative Hierarchical Clustering on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to form.
        random_state (int): Random seed for reproducibility.
        n_components (int): Number of PCA components (used if use_pca_data).
        use_pca_data (bool): Whether to apply PCA before clustering.

    Returns:
        sklearn.cluster.AgglomerativeClustering: Fitted clustering instance.
        numpy.ndarray: Data used for fitting (PCA-reduced or original).
        float: Silhouette score of the clustering (None if n_clusters=1).
    """
    if use_pca_data:
        X_fit, _ = Apply_PCA(X, n_components=n_components,
                             random_state=random_state)
    else:
        X_fit = X

    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    model.fit(X_fit)

    if n_clusters > 1:
        score = metrics.silhouette_score(X_fit, model.labels_)
    else:
        score = None

    return model, X_fit, score
