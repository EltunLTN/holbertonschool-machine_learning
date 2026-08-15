#!/usr/bin/env python3
"""
DeepNeuralNetwork class for binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Class constructor.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list or empty.
            TypeError: If elements in layers are not positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            layer_size = layers[l]
            prev_layer_size = nx if l == 0 else layers[l - 1]
            # He et al. initialization
            self.weights['W' + str(l + 1)] = (np.random.randn(layer_size, prev_layer_size) *
                                              np.sqrt(2 / prev_layer_size))
            self.weights['b' + str(l + 1)] = np.zeros((layer_size, 1))
