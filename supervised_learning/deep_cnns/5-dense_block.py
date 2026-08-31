#!/usr/bin/env python3
"""Builds a dense block as described in Densely Connected
Convolutional Networks (DenseNet)"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block

    Uses the bottleneck layers used for DenseNet-B
    All weights use he normal initialization with seed 0
    Every convolution is preceded by Batch Normalization and ReLU

    Returns: the concatenated output of each layer within the Dense
    Block, and the number of filters within the concatenated outputs
    """
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        # Bottleneck layer: BN -> ReLU -> 1x1 conv (4 * growth_rate filters)
        bn1 = K.layers.BatchNormalization()(X)
        relu1 = K.layers.Activation('relu')(bn1)
        bottleneck = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            kernel_initializer=init
        )(relu1)

        # Composite layer: BN -> ReLU -> 3x3 conv (growth_rate filters)
        bn2 = K.layers.BatchNormalization()(bottleneck)
        relu2 = K.layers.Activation('relu')(bn2)
        conv = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=init
        )(relu2)

        # Concatenate the new feature maps with all previous ones
        X = K.layers.Concatenate(axis=-1)([X, conv])
        nb_filters += growth_rate

    return X, nb_filters
