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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(self.__L):
            if type(layers[l]) is not int or layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")

            if l == 0:
                prev = nx
            else:
                prev = layers[l - 1]

            self.__weights["W" + str(l + 1)] = (
                np.random.randn(layers[l], prev) * np.sqrt(2 / prev)
            )
            self.__weights["b" + str(l + 1)] = np.zeros((layers[l], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        X: numpy.ndarray with shape (nx, m) containing the input data
        Returns: the output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X

        for l in range(1, self.__L + 1):
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]
            A_prev = self.__cache["A" + str(l - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A" + str(l)] = A

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Y: numpy.ndarray shape (1, m) - correct labels
        A: numpy.ndarray shape (1, m) - activated output
        Returns: the cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        X: numpy.ndarray shape (nx, m) - input data
        Y: numpy.ndarray shape (1, m) - correct labels
        Returns: the neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Y: numpy.ndarray shape (1, m) - correct labels
        cache: dictionary containing all intermediary values of the network
        alpha: learning rate
        Updates the private attribute __weights
        """
        m = Y.shape[1]
        L = self.__L

        dZ = cache["A" + str(L)] - Y

        for l in range(L, 0, -1):
            A_prev = cache["A" + str(l - 1)]
            W = self.__weights["W" + str(l)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network

        X: numpy.ndarray shape (nx, m) - input data
        Y: numpy.ndarray shape (1, m) - correct labels
        iterations: number of iterations to train over
        alpha: learning rate
        Returns: the evaluation of the training data after training
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
