#!/usr/bin/env python3
"""Module that calculates the determinant of a matrix."""


def determinant(matrix):
    """Calculate the determinant of a matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0
    for j in range(len(matrix)):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        total += ((-1) ** j) * matrix[0][j] * determinant(minor)
    return total
