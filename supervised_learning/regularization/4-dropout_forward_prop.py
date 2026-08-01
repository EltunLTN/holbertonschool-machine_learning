#!/usr/bin/env python3
"""Forward propagation with Dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    X: numpy.ndarray of shape (nx, m) containing the input data
    weights: dict of weights and biases of the network
    L: number of layers in the network
    keep_prob: probability that a node will be kept

    All layers except the last use tanh; the last uses softmax.

    Returns: a dict containing the outputs of each layer and the
    dropout mask used on each layer.
    """
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]

        Z = np.matmul(W, A_prev) + b

        if layer == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['D' + str(layer)] = D

        cache['A' + str(layer)] = A

    return cache
