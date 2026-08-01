#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout in a neural network.
"""

import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with dropout.

    Parameters:
    - X: numpy.ndarray of shape (nx, m), input data
    - weights: dict, weights and biases of the network
    - L: int, number of layers in the network
    - keep_prob: float, probability that a node will be kept

    Returns:
    - cache: dict containing the outputs of each layer and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        A_prev = cache['A' + str(l - 1)]

        Z = np.matmul(W, A_prev) + b

        if l != L:
            # Hidden layers: tanh activation
            A = np.tanh(Z)

            # Dropout mask
            D = (np.random.rand(*A.shape) < keep_prob).astype(int)

            # Apply mask and scale activations
            A *= D
            A /= keep_prob

            cache['D' + str(l)] = D
        else:
            # Output layer: softmax activation
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # for numerical stability
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(l)] = A

    return cache
