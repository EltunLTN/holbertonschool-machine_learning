#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Class constructor

        nx is the number of input features
        layers is a list representing the number of nodes in each
        layer of the network
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

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1]))
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases of the network"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        X is a numpy.ndarray with shape (nx, m) that contains the
        input data
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A_prev = self.__cache['A' + str(i)]
            z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache['A' + str(i + 1)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        X is a numpy.ndarray with shape (nx, m) that contains the
        input data
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        """
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        cache is a dictionary containing all the intermediary values
        of the network
        alpha is the learning rate
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()

        A_L = cache['A' + str(L)]
        dz = A_L - Y

        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = weights['W' + str(i)]

            dW = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                dz = np.matmul(W.T, dz) * (A_prev * (1 - A_prev))

            self.__weights['W' + str(i)] = W - alpha * dW
            self.__weights['b' + str(i)] = (
                weights['b' + str(i)] - alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network

        X is a numpy.ndarray with shape (nx, m) that contains the
        input data
        Y is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations is the number of iterations to train over
        alpha is the learning rate
        verbose is a boolean that defines whether or not to print
        information about the training
        graph is a boolean that defines whether or not to graph
        information about the training once the training has completed
        step is the number of iterations between each print/graph point
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
