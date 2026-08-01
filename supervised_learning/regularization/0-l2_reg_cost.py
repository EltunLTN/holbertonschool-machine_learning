#!/usr/bin/env python3
"""
Function to calculate the cost of a neural network with L2 regularization.
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    - cost: numpy.ndarray or float, the cost of the network without L2 regularization
    - lambtha: float, the regularization parameter
    - weights: dict, dictionary of the weights and biases of the neural network
    - L: int, the number of layers in the neural network
    - m: int, the number of data points used

    Returns:
    - float, the cost of the network accounting for L2 regularization
    """
    sum_weights_squared = 0
    for l in range(1, L + 1):
        W = weights.get('W' + str(l))
        if W is not None:
            sum_weights_squared += np.sum(np.square(W))
    l2_term = (lambtha / (2 * m)) * sum_weights_squared
    return float(cost + l2_term)
