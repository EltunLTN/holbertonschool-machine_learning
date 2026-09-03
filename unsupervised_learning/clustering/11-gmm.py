#!/usr/bin/env python3
import sklearn.mixture


def gmm(X, k):
    """Calculates a Gaussian Mixture Model using scikit-learn."""
    if not hasattr(X, 'ndim') or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None

    model = sklearn.mixture.GaussianMixture(
        n_components=k, random_state=0)
    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
