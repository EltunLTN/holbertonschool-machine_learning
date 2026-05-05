#!/usr/bin/env python3
"""Normal distribution module"""


class Normal:
    """Class that represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def pdf(self, x):
        """
        Calculates the PDF for a given x-value

        Args:
            x (float): x-value

        Returns:
            float: PDF value
        """

        if self.stddev <= 0:
            return 0

        e = 2.7182818285
        pi = 3.1415926536

        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        denominator = self.stddev * (2 * pi) ** 0.5

        return (1 / denominator) * (e ** exponent)
