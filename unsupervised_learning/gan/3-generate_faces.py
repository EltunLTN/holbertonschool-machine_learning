#!/usr/bin/env python3
"""Module that defines convolutional Generator and Discriminator for GANs."""

from tensorflow.keras import layers, models


def convolutional_GenDiscr():
    """Builds and returns a convolutional Generator and Discriminator.

    Returns:
        tuple: (generator, discriminator) Keras models.
    """
    def get_generator():
        model = models.Sequential([
            layers.Input(shape=(16,)),
            layers.Dense(2048, activation='tanh'),
            layers.Reshape((2, 2, 512)),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('tanh'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(16, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('tanh'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('tanh')
        ], name='generator')
        return model

    def get_discriminator():
        model = models.Sequential([
            layers.Input(shape=(16, 16, 1)),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Activation('tanh'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Activation('tanh'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Activation('tanh'),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Activation('tanh'),
            layers.Flatten(),
            layers.Dense(1, activation='tanh')
        ], name='discriminator')
        return model

    return get_generator(), get_discriminator()
