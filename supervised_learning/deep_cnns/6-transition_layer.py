#!/usr/bin/env python3
"""Builds a transition layer as described in Densely Connected
Convolutional Networks (DenseNet)"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    Implements compression as used in DenseNet-C
    All weights use he normal initialization with seed 0
    The convolution is preceded by Batch Normalization and ReLU

    Returns: the output of the transition layer, and the number of
    filters within the output
    """
    init = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    bn = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=init
    )(relu)

    avg_pool = K.layers.AveragePooling2D(
        pool_size=2,
        strides=2,
        padding='same'
    )(conv)

    return avg_pool, nb_filters
