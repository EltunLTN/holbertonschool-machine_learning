#!/usr/bin/env python3
"""Module that calculates the precision for each class in a
confusion matrix.
"""
import numpy as np


def precision(confusion):
    """Calculate the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes), where row indices represent the
            correct labels and column indices represent the
            predicted labels.

    Returns:
        numpy.ndarray: array of shape (classes,) containing the
            precision of each class.
    """
    true_positives = np.diag(confusion)
    predicted_totals = np.sum(confusion, axis=0)
    return true_positives / predicted_totals
