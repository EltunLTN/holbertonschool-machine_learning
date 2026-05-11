#!/usr/bin/env python3
"""Posterior Probability"""

import numpy as np
from math import factorial


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data.

    Args:
        x  - number of patients that develop severe side effects
        n  - total number of patients observed
        P  - 1D numpy.ndarray of hypothetical probabilities
        Pr - 1D numpy.ndarray of prior beliefs about P

    Returns:
        Posterior probability of each probability in P given x and n
    """

    # --- Validation ---

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")

    # --- Calculation ---

    # Binomial coefficient: C(n, x)
    coefficient = factorial(n) / (factorial(x) * factorial(n - x))

    # Likelihood: P(x | p) for each p in P
    likelihood = coefficient * (P ** x) * ((1 - P) ** (n - x))

    # Intersection: P(x, p) = likelihood * prior
    intersection = likelihood * Pr

    # Marginal: total probability P(x)
    marginal = np.sum(intersection)

    # Posterior: P(p | x) = P(x, p) / P(x)   [Bayes' theorem]
    return intersection / marginal
