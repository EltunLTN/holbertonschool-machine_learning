#!/usr/bin/env python3
"""Makes predictions using a Keras neural network."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Return the predictions produced by a neural network."""
    return network.predict(
        data,
        verbose=verbose
    )
