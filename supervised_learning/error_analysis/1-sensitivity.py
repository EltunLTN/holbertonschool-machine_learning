#!/usr/bin/env python3
"""Module that calculates the sensitivity for each class in a
confusion matrix.
"""
import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes), where row indices represent the
            correct labels and column indices represent the
            predicted labels.

    Returns:
        numpy.ndarray: array of shape (classes,) containing the
            sensitivity of each class.
    """
    true_positives = np.diag(confusion)
    actual_totals = np.sum(confusion, axis=1)
    return true_positives / actual_totals
