#!/usr/bin/env python3
"""Evaluates a Keras neural network."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Evaluate a neural network using testing data."""
    return network.evaluate(
        data,
        labels,
        verbose=verbose
    )
