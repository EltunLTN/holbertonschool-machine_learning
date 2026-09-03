#!/usr/bin/env python3
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5,
        verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin <= 0 or kmin > n:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None
    if kmax == kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = np.arange(kmin, kmax + 1)
    l = np.empty(len(ks))
    b = np.empty(len(ks))
    best_k = None
    best_result = None
    best_bic = np.inf

    for i, k in enumerate(ks):
        result = expectation_maximization(
            X, int(k), iterations, tol, verbose)
        if result[0] is None:
            return None, None, None, None
        pi, m, S, _, likelihood = result
        p = k * d + k * d * (d + 1) / 2 + k - 1
        bic = p * np.log(n) - 2 * likelihood
        l[i] = likelihood
        b[i] = bic
        if bic < best_bic:
            best_bic = bic
            best_k = int(k)
            best_result = (pi, m, S)

    return best_k, best_result, l, b
