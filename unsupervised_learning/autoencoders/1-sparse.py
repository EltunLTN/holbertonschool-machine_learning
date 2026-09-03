#!/usr/bin/env python3
"""Sparse Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in the encoder
        latent_dims: integer, dimensions of the latent space representation
        lambtha: L1 regularization parameter for the encoded output

    Returns:
        encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    reg = keras.regularizers.l1(lambtha)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=reg)(x)
    encoder = keras.Model(inputs, latent)

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs, auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
