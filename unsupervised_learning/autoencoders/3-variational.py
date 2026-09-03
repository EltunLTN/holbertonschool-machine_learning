#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder

    Args:
        input_dims: integer, dimensions of the model input
        hidden_layers: list of nodes for each hidden layer in the encoder
        latent_dims: integer, dimensions of the latent space representation

    Returns:
        encoder, decoder, auto
    """
    def sampling(args):
        """Reparameterization trick: sample z from mu and log_var"""
        mu, log_var = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mu + K.exp(log_var / 2) * epsilon

    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,))([mu, log_var])

    encoder = keras.Model(inputs, [z, mu, log_var])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs)

    z_out, mu_out, log_var_out = encoder(inputs)
    auto_outputs = decoder(z_out)
    auto = keras.Model(inputs, auto_outputs)

    def vae_loss(y_true, y_pred):
        """Computes reconstruction loss plus KL divergence loss"""
        reconstruction_loss = keras.losses.binary_crossentropy(
            y_true, y_pred)
        reconstruction_loss *= input_dims
        kl_loss = 1 + log_var_out - K.square(mu_out) - K.exp(log_var_out)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
