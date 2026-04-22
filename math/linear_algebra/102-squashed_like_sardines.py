#!/usr/bin/env python3
"""
    Concatenates two matrices along a given axis
    """


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a given axis
    """

    # Helper: get shape of matrix
    def get_shape(mat):
        shape = []
        while isinstance(mat, list):
            shape.append(len(mat))
            mat = mat[0]
        return shape

    # Helper: recursive concatenation
    def concat(m1, m2, ax):
        if ax == 0:
            return m1 + m2
        return [concat(sub1, sub2, ax - 1) for sub1, sub2 in zip(m1, m2)]

    # Get shapes
    shape1 = get_shape(mat1)
    shape2 = get_shape(mat2)

    # Must have same number of dimensions
    if len(shape1) != len(shape2):
        return None

    # Check compatibility
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    # Perform concatenation
    try:
        return concat(mat1, mat2, axis)
    except Exception:
        return None
