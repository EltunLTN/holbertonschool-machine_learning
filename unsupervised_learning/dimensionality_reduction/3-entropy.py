#!/usr/bin/env python3
"""Defines a function that calculates the Shannon entropy and P
affinities relative to a data point."""
import numpy as np


def HP(Di, beta):
    """Calculates the Shannon entropy and P affinities relative to
    a data point.

    Args:
        Di (numpy.ndarray): array of shape (n - 1,) containing the
            pairwise distances between a data point and all other
            points except itself, where n is the number of data
            points.
        beta (numpy.ndarray): array of shape (1,) containing the
            beta value for the Gaussian distribution.

    Returns:
        tuple: (Hi, Pi)
            Hi (float): the Shannon entropy of the points.
            Pi (numpy.ndarray): array of shape (n - 1,) containing
                the P affinities of the points.
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
