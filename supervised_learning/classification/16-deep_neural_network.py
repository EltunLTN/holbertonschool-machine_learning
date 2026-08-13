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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        # Initialize weights and biases
        # He et al. initialization: np.random.randn(...) * np.sqrt(2 / prev_layer_nodes)
        for l in range(self.L):
            prev_nodes = nx if l == 0 else layers[l - 1]
            curr_nodes = layers[l]
            
            self.weights[f'W{l + 1}'] = np.random.randn(curr_nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            self.weights[f'b{l + 1}'] = np.zeros((curr_nodes, 1))

        @property
        def L(self):
            return self.__L

        @L.setter
        def L(self, value):
            self.__L = value

        @property
        def cache(self):
            return self.__cache

        @cache.setter
        def cache(self, value):
            self.__cache = value

        @property
        def weights(self):
            return self.__weights

        @weights.setter
        def weights(self, value):
            self.__weights = value
