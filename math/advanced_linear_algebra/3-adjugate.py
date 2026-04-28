#!/usr/bin/env python3
"""Module that calculates the minor matrix of a matrix."""


def determinant(matrix):
    """Calculate the determinant of a matrix."""
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0
    for j in range(len(matrix)):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        total += ((-1) ** j) * matrix[0][j] * determinant(minor)
    return total


def minor(matrix):
    """Calculate the minor matrix of a matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            sub = [r[:j] + r[j+1:] for r in (matrix[:i] + matrix[i+1:])]
            row.append(determinant(sub))
        result.append(row)
    return result


def cofactor(matrix):
    """Calculate the cofactor matrix of a matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    minor_matrix = minor(matrix)
    return [[minor_matrix[i][j] * ((-1) ** (i + j))
             for j in range(len(matrix))]
            for i in range(len(matrix))]


def adjugate(matrix):
    """Calculate the adjugate matrix of a matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    cofactor_matrix = cofactor(matrix)
    return [[cofactor_matrix[j][i]
             for j in range(len(matrix))]
            for i in range(len(matrix))]
