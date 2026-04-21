#!/usr/bin/env python3
"""Module that calculates the shape of a matrix."""


def matrix_shape(matrix):
    """Calculate and return the shape of a matrix as a list of integers."""
    shape = []
    current = matrix
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0]
    return shape
