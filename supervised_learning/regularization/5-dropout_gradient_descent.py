#!/usr/bin/env python3
"""Gradient descent with Dropout."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights: dict of weights and biases of the network
    cache: dict of outputs and dropout masks of each layer
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers of the network

    All layers use tanh except the last, which uses softmax.
    Weights are updated in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = (dA_prev * cache['D' + str(layer - 1)]) / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
