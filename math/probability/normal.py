#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution

        Parameters:
        data (list): data to estimate mean and stddev
        mean (float): mean of the distribution
        stddev (float): standard deviation
        """

        if data is None:
            # Validate stddev
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Compute mean
            self.mean = float(sum(data) / len(data))

            # Compute standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)
