#!/usr/bin/env python3
"""
DeepNeuralNetwork class implementation with private attributes and
He initialization for binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Class constructor.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list or is empty.
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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            layer_input = nx if l == 0 else layers[l - 1]
            self.__weights['W' + str(l + 1)] = (
                np.random.randn(layers[l], layer_input) * np.sqrt(2 / layer_input)
            )
            self.__weights['b' + str(l + 1)] = np.zeros((layers[l], 1))

    @property
    def L(self):
        """Number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary to hold all intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary to hold all weights and biases of the network."""
        return self.__weights
