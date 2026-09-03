#!/usr/bin/env python3
"""
Policy gradient module for reinforcement learning.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy probability distribution given a state matrix and weights.

    Args:
        matrix (numpy.ndarray): The state matrix (observation).
        weight (numpy.ndarray): The weight matrix.

    Returns:
        numpy.ndarray: The computed policy probabilities.
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp, axis=1, keepdims=True)
