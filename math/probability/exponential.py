#!/usr/bin/env python3
"""Exponential distribution module"""


class Exponential:
    """Class that represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution

        Parameters:
        data (list): data to estimate lambtha
        lambtha (float): expected number of occurrences
        """

        if data is None:
            # Validate lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # For exponential: lambtha = 1 / mean
            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)
