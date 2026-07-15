#!/usr/bin/env python3
"""Functions for saving and loading Keras models."""

import tensorflow.keras as K


def save_model(network, filename):
    """Save an entire Keras model to a file."""
    network.save(filename)


def load_model(filename):
    """Load and return an entire Keras model from a file."""
    return K.models.load_model(filename)
