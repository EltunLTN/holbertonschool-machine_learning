#!/usr/bin/env python3
"""Trains and optionally saves the best Keras model."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """Train a model using optional Keras callbacks."""
    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience
        )
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """Calculate the learning rate for the current epoch."""
            return alpha / (1 + decay_rate * epoch)

        learning_rate = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )
        callbacks.append(learning_rate)

    if save_best and validation_data is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        )
        callbacks.append(checkpoint)

    return network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
