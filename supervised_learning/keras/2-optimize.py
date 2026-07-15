#!/usr/bin/env python3
"""Configures Adam optimization for a Keras model."""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Compile a model using Adam and categorical crossentropy."""
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
