#!/usr/bin/env python3
"""Module that slices a matrix along specific axes."""
import numpy as np


def np_slice(matrix, axes={}):
    """Slice a matrix along specific axes."""
    slices = [slice(None)] * matrix.ndim
    for axis, s in axes.items():
        slices[axis] = slice(*s)
    return matrix[tuple(slices)]
