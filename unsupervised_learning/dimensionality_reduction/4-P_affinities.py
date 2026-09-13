#!/usr/bin/env python3
"""Defines a function that calculates the symmetric P affinities of
a dataset for t-SNE."""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculates the symmetric P affinities of a dataset.

    Args:
        X (numpy.ndarray): array of shape (n, d) containing the
            dataset to be transformed by t-SNE, where n is the
            number of data points and d is the number of dimensions
            in each point.
        tol (float): the maximum tolerance allowed (inclusive) for
            the difference in Shannon entropy from perplexity for
            all Gaussian distributions.
        perplexity (float): the perplexity that all Gaussian
            distributions should have.

    Returns:
        numpy.ndarray: P, array of shape (n, n) containing the
            symmetric P affinities.
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low = None
        high = None
        beta = betas[i].copy()

        Di = np.delete(D[i], i)
        Hi, Pi = HP(Di, beta)
        Hdiff = Hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = beta[0]
                if high is None:
                    beta[0] = beta[0] * 2
                else:
                    beta[0] = (beta[0] + high) / 2
            else:
                high = beta[0]
                if low is None:
                    beta[0] = beta[0] / 2
                else:
                    beta[0] = (beta[0] + low) / 2

            Hi, Pi = HP(Di, beta)
            Hdiff = Hi - H

        betas[i] = beta
        P[i, np.arange(n) != i] = Pi

    P = (P + P.T) / (2 * n)
    return P
