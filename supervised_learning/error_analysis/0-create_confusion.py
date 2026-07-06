#!/usr/bin/env python3
"""Module that creates a confusion matrix from one-hot labels and logits."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix.

    Args:
        labels (numpy.ndarray): one-hot array of shape (m, classes)
            containing the correct labels for each data point.
        logits (numpy.ndarray): one-hot array of shape (m, classes)
            containing the predicted labels for each data point.

    Returns:
        numpy.ndarray: confusion matrix of shape (classes, classes),
            where row indices represent the correct labels and column
            indices represent the predicted labels.
    """
    return np.matmul(labels.T, logits)
 