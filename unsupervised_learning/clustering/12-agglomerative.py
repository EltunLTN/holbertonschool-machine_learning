#!/usr/bin/env python3
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs Ward agglomerative clustering and displays a dendrogram."""
    if not hasattr(X, 'ndim') or X.ndim != 2:
        return None
    if not isinstance(dist, (int, float)) or dist <= 0:
        return None

    linkage = scipy.cluster.hierarchy.ward(X)
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion='distance') - 1
    return clss
