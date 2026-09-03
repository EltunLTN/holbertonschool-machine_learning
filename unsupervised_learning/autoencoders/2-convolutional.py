#!/usr/bin/env python3
"""Convolutional Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    Args:
        input_dims: tuple of integers, dimensions of the model input
        filters: list of filters for each conv layer in the encoder
        latent_dims: tuple of integers, dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs, x)

    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs
    rev_filters = list(reversed(filters))
    for i, f in enumerate(rev_filters[:-1]):
        x = keras.layers.Conv2D(
            f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        rev_filters[-1], (3, 3), padding='valid', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
