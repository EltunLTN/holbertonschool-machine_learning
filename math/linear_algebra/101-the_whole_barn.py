#!/usr/bin/env python3
"""Module that adds two matrices of any dimension."""


def add_matrices(mat1, mat2):
    """Add two matrices of any dimension element-wise."""
    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        result = [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
        if None in result:
            return None
        return result
    return mat1 + mat2
