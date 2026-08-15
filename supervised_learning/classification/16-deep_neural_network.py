#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network for binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor

        nx: number of input features
        layers: list representing number of nodes in each layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if type(layers[l]) is not int or layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")

            if l == 0:
                prev = nx
            else:
                prev = layers[l - 1]

            self.weights["W" + str(l + 1)] = (
                np.random.randn(layers[l], prev) * np.sqrt(2 / prev)
            )
            self.weights["b" + str(l + 1)] = np.zeros((layers[l], 1))
