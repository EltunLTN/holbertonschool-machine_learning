#!/usr/bin/env python3
"""Module that calculates the specificity for each class in a
confusion matrix.
"""
import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): confusion matrix of shape
            (classes, classes), where row indices represent the
            correct labels and column indices represent the
            predicted labels.

    Returns:
        numpy.ndarray: array of shape (classes,) containing the
            specificity of each class.
    """
    total = np.sum(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    true_positives = np.diag(confusion)
    false_positives = col_sums - true_positives
    false_negatives = row_sums - true_positives
    true_negatives = total - true_positives - false_positives \
        - false_negatives
    return true_negatives / (true_negatives + false_positives)
