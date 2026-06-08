#!/usr/bin/env python3
"""Module for performing PCA dimensionality reduction using Scikit-learn."""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Perform Principal Component Analysis on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int | float | None): Number of components to keep.
            int: exact number of principal components.
            float (0-1): minimum fraction of variance to preserve.
            None: keep all components.
        random_state (int): Random seed for reproducibility.

    Returns:
        numpy.ndarray: Data transformed into principal component space.
        sklearn.decomposition.PCA: Fitted PCA instance.
    """
    pca = decomposition.PCA(
        n_components=n_components,
        random_state=random_state
    )
    X_pca = pca.fit_transform(X)
    return X_pca, pca
