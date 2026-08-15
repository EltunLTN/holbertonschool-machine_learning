#!/usr/bin/env python3
"""Defines a deep neural network performing binary/multiclass classification"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """Deep Neural Network for binary/multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Class constructor

        nx: number of input features
        layers: list representing number of nodes in each layer
        activation: activation function for hidden layers ('sig' or 'tanh')
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        return self.__activation

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

            if l == self.__L:
                # Output layer: softmax (multiclass)
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Hidden layers: sig or tanh
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)

            self.__cache["A" + str(l)] = A

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using categorical cross-entropy

        Y: numpy.ndarray shape (classes, m) - one-hot correct labels
        A: numpy.ndarray shape (classes, m) - activated output
        Returns: the cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        X: numpy.ndarray shape (nx, m) - input data
        Y: numpy.ndarray shape (classes, m) - one-hot correct labels
        Returns: the neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Y: numpy.ndarray shape (classes, m) - one-hot correct labels
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
                if self.__activation == 'sig':
                    deriv = A_prev * (1 - A_prev)
                else:
                    deriv = 1 - A_prev ** 2
                dZ = np.matmul(W.T, dZ) * deriv

            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network

        X: numpy.ndarray shape (nx, m) - input data
        Y: numpy.ndarray shape (classes, m) - one-hot correct labels
        iterations: number of iterations to train over
        alpha: learning rate
        verbose: whether to print training info
        graph: whether to plot training info
        step: interval for verbose/graph updates
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
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
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

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        filename: file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        filename: file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn't exist
        """
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
