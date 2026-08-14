import numpy as np

class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Class constructor.

        Parameters:
        nx (int): Number of input features.
        layers (list): List representing the number of nodes in each layer.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        TypeError: If layers is not a list or is empty.
        TypeError: If elements in layers are not all positive integers.
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}       # dictionary to hold intermediary values
        self.weights = {}     # dictionary to hold weights and biases

        # Initialize weights and biases using He et al. method
        for l in range(self.L):
            layer_size = layers[l]
            prev_layer_size = nx if l == 0 else layers[l - 1]

            # He initialization for weights
            self.weights[f"W{l + 1}"] = (np.random.randn(layer_size, prev_layer_size) *
                                        np.sqrt(2 / prev_layer_size))
            # Biases initialized to zeros
            self.weights[f"b{l + 1}"] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Number of layers in the neural network."""
        return self.__L

    @L.setter
    def L(self, value):
        self.__L = value

    @property
    def cache(self):
        """Dictionary to hold all intermediary values of the network."""
        return self.__cache

    @cache.setter
    def cache(self, value):
        self.__cache = value

    @property
    def weights(self):
        """Dictionary to hold all weights and biases of the network."""
        return self.__weights

    @weights.setter
    def weights(self, value):
        self.__weights = value
