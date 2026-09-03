#!/usr/bin/env python3
import numpy as np


def maximization(X, g):
    """Calculates the maximization step for a GMM."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n or k == 0:
        return None, None, None

    nk = np.sum(g, axis=1)
    if np.any(nk <= 0) or not np.all(np.isfinite(nk)):
        return None, None, None

    pi = nk / n
    m = np.dot(g, X) / nk[:, None]
    diff = X[None, :, :] - m[:, None, :]
    S = np.einsum('kn,kni,knj->kij', g, diff, diff) / nk[:, None, None]

    return pi, m, S
