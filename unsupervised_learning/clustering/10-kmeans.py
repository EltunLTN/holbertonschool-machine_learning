#!/usr/bin/env python3
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means clustering using scikit-learn."""
    if not hasattr(X, 'ndim') or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    model = sklearn.cluster.KMeans(
        n_clusters=k, random_state=0, n_init=10)
    clss = model.fit_predict(X)
    return model.cluster_centers_, clss
