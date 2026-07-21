#!/usr/bin/env python3
"""Defines a function that calculates the weighted moving average
of a data set"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: the list of data to calculate the moving average of
        beta: the weight used for the moving average

    Returns:
        a list containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        bias_correction = 1 - beta ** (i + 1)
        moving_averages.append(v / bias_correction)

    return moving_averages
