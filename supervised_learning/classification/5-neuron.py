#!/usr/bin/env python3
"""Gradient descent module for a neuron performing binary classification."""
import numpy as np


class Neuron:
    """Attributes:
    __W (numpy.ndarray): The weights vector for the neuron.
    __b (float): The bias for the neuron.
    __A (float): The activated output of the neuron (prediction).
"""

def __init__(self, nx):
    """
    Initialize the neuron.

    Args:
        nx (int): Number of input features to the neuron.

    Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
    """
    if not isinstance(nx, int):
        raise TypeError("nx must be an integer")
    if nx < 1:
        raise ValueError("nx must be a positive integer")
    self.__W = np.random.randn(1, nx)
    self.__b = 0
    self.__A = 0

@property
def W(self):
    """
    Getter for the weights vector.

    Returns:
        numpy.ndarray: Weights vector.
    """
    return self.__W

@property
def b(self):
    """
    Getter for the bias.

    Returns:
        float: Bias.
    """
    return self.__b

@property
def A(self):
    """
    Getter for the activated output.

    Returns:
        float: Activated output (prediction).
    """
    return self.__A

def forward_prop(self, X):
    """
    Calculate the forward propagation of the neuron.

    Args:
        X (numpy.ndarray): Input data of shape (nx, m).

    Returns:
        numpy.ndarray: Activated output of the neuron.
    """
    z = np.dot(self.__W, X) + self.__b
    self.__A = 1 / (1 + np.exp(-z))
    return self.__A

def gradient_descent(self, X, Y, A, alpha=0.05):
    """
    Perform one pass of gradient descent on the neuron to update weights and bias.

    Args:
        X (numpy.ndarray): Input data of shape (nx, m).
        Y (numpy.ndarray): Correct labels of shape (1, m).
        A (numpy.ndarray): Activated output of shape (1, m).
        alpha (float): Learning rate.

    Updates:
        __W and __b attributes of the neuron.
    """
    m = Y.shape[1]
    dZ = A - Y
    dW = np.dot(dZ, X.T) / m
    db = np.sum(dZ) / m
    self.__W = self.__W - alpha * dW
    self.__b = self.__b - alpha * db
