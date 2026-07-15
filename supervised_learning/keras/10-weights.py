#!/usr/bin/env python3
"""Functions for saving and loading Keras model weights."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save a Keras model's weights to a file."""
    network.save_weights(
        filename,
        save_format=save_format
    )


def load_weights(network, filename):
    """Load weights into a Keras model."""
    network.load_weights(filename)
