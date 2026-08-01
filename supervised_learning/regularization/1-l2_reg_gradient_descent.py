#!/usr/bin/env python3
"""
Updates weights and biases using gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient 
    descent with L2 regularization.

    Parameters:
    - Y: numpy.ndarray of shape (classes, m), correct labels in one-hot encoding
    - weights: dict, weights and biases of the neural network
    - cache: dict, outputs of each layer of the neural network
    - alpha: float, learning rate
    - lambtha: float, L2 regularization parameter
    - L: int, number of layers in the network

    Updates weights and biases in place.
    """
    m = Y.shape[1]

    # Gradient of output layer (softmax + cross-entropy)
    dZ = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        # Gradient with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = b - alpha * db

        if l > 1:
            # Backpropagate through tanh activation: derivative = 1 - A^2
            A_prev_layer = cache['A' + str(l - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev_layer))
