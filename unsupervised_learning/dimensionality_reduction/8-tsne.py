#!/usr/bin/env python3
"""Defines a function that performs a t-SNE transformation."""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Performs a t-SNE transformation.

    Args:
        X (numpy.ndarray): array of shape (n, d) containing the
            dataset to be transformed by t-SNE, where n is the
            number of data points and d is the number of dimensions
            in each point.
        ndims (int): the new dimensional representation of X.
        idims (int): the intermediate dimensional representation of
            X after PCA.
        perplexity (float): the perplexity.
        iterations (int): the number of iterations.
        lr (float): the learning rate.

    Returns:
        numpy.ndarray: Y, array of shape (n, ndims) containing the
            optimized low dimensional transformation of X.
    """
    n, d = X.shape
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    P = P * 4
    Y = np.random.randn(n, ndims)
    Y_prev1 = Y.copy()
    Y_prev2 = Y.copy()

    for i in range(iterations):
        dY, Q = grads(Y, P)

        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        Y = Y - lr * dY + momentum * (Y_prev1 - Y_prev2)
        Y = Y - np.mean(Y, axis=0)

        Y_prev2 = Y_prev1.copy()
        Y_prev1 = Y.copy()

        if i == 99:
            P = P / 4

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print('Cost at iteration {}: {}'.format(i + 1, C))

    return Y
