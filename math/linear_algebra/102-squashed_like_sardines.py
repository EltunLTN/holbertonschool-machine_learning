#!/usr/bin/env python3
"""Module that concatenates two matrices along a specific axis."""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""
    if axis == 0:
        if isinstance(mat1[0], list):
            if len(mat1[0]) != len(mat2[0]):
                return None
        return mat1 + mat2
    if isinstance(mat1[0], list):
        if len(mat1) != len(mat2):
            return None
        result = [cat_matrices(mat1[i], mat2[i], axis - 1)
                  for i in range(len(mat1))]
        if None in result:
            return None
        return result
    return None
