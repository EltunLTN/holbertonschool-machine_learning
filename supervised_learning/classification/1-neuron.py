#!/usr/bin/env python3
"privatice neuron"
import numpy as np


class Neuron:
    """private neuron"""

    def __init__(self, nx):
        """private neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise TypeError("nx must be positive")
        self.__W = np.random.randn()
        self.__b = 0
        self.__a = 0
