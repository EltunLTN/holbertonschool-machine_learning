#!/usr/bin/env python3
"""L2 regularization cost."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    cost: cost of the network without L2 regularization
    lambtha: regularization parameter
    weights: dict of the weights and biases (numpy.ndarrays) of the
             network
    L: number of layers in the network
    m: number of data points used

    Returns: the cost of the network accounting for L2 regularization.
    """
    weights_sum = 0
    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        weights_sum += np.linalg.norm(W) ** 2

    l2_cost = cost + (lambtha / (2 * m)) * weights_sum
    return l2_cost
