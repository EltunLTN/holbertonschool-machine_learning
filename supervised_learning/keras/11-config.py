#!/usr/bin/env python3
"""Functions for saving and loading a Keras model configuration."""

import tensorflow.keras as K


def save_config(network, filename):
    """Save a model's configuration in JSON format."""
    with open(filename, "w") as file:
        file.write(network.to_json())


def load_config(filename):
    """Load a Keras model from a JSON configuration file."""
    with open(filename, "r") as file:
        configuration = file.read()

    return K.models.model_from_json(configuration)
