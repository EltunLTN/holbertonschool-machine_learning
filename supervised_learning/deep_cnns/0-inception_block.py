#!/usr/bin/env python3
"""Builds an inception block (Going Deeper with Convolutions, 2014)"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3, F5R, F5, FPP:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution
            after the max pooling

    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 conv
    conv_1x1 = K.layers.Conv2D(
        filters=F1, kernel_size=1, padding='same',
        activation='relu')(A_prev)

    # Branch 2: 1x1 conv -> 3x3 conv
    conv_3x3_reduce = K.layers.Conv2D(
        filters=F3R, kernel_size=1, padding='same',
        activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(
        filters=F3, kernel_size=3, padding='same',
        activation='relu')(conv_3x3_reduce)

    # Branch 3: 1x1 conv -> 5x5 conv
    conv_5x5_reduce = K.layers.Conv2D(
        filters=F5R, kernel_size=1, padding='same',
        activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(
        filters=F5, kernel_size=5, padding='same',
        activation='relu')(conv_5x5_reduce)

    # Branch 4: 3x3 max pool -> 1x1 conv
    pool = K.layers.MaxPooling2D(
        pool_size=3, strides=1, padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(
        filters=FPP, kernel_size=1, padding='same',
        activation='relu')(pool)

    # Concatenate all branches along the channel axis
    output = K.layers.Concatenate(axis=-1)(
        [conv_1x1, conv_3x3, conv_5x5, pool_proj])

    return output
