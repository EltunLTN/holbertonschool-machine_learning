#!/usr/bin/env python3
"""Binomial distribution module"""


class Binomial:
    """Class that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution

        Parameters:
        data (list): data to estimate n and p
        n (int): number of trials
        p (float): probability of success
        """

        if data is None:
            # Validate n and p
            if n <= 0:
                raise ValueError("n must be a positive value")

            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Compute mean
            mean = sum(data) / len(data)

            # Compute variance
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # Estimate p first
            p = 1 - (variance / mean)

            # Estimate n
            n = mean / p

            # Round n (important!)
            self.n = int(round(n))

            # Recalculate p using rounded n
            self.p = float(mean / self.n)
