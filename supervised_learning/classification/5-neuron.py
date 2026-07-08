#!/usr/bin/env python3
"""Gradient descent module for a neuron performing binary classification."""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize the neuron.

        Parameters:
        nx (int): Number of input features to the neuron.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
        numpy.ndarray: Activated output of the neuron (1, m).
        """
        z = np.dot(self.W, X) + self.b
        self.A = 1 / (1 + np.exp(-z))
        return self.A

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.

        Parameters:
        Y (numpy.ndarray): Correct labels of shape (1, m).
        A (numpy.ndarray): Activated output of shape (1, m).

        Returns:
        float: The cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculate one pass of gradient descent on the neuron.

        Parameters:
        X (numpy.ndarray): Input data of shape (nx, m).
        Y (numpy.ndarray): Correct labels of shape (1, m).
        A (numpy.ndarray): Activated output of shape (1, m).
        alpha (float): The learning rate.

        Returns:
        None
        """
        m = X.shape[1]
        dz = A - Y
        dw = (1 / m) * np.dot(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.W -= alpha * dw
        self.b -= alpha * db
