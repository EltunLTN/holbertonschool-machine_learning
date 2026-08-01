#!/usr/bin/env python3
"""
Creates a neural network layer in TensorFlow with L2 regularization.
"""

import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a layer with L2 regularization.

    Parameters:
    - prev: tensor, output of the previous layer
    - n: int, number of nodes in the new layer
    - activation: activation function to use on the layer
    - lambtha: float, L2 regularization parameter

    Returns:
    - tensor, output of the new layer
    """
    kernel_regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_regularizer=kernel_regularizer)(prev)
    return layer
