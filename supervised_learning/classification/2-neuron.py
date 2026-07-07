#!/usr/bin/env python3
"""Defines a single neuron performing binary classificatio"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        nx is the number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        Updates the private attribute __A using sigmoid activation
        Returns the private attribute __A
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))  # Sigmoid activation function
        return self.__A
