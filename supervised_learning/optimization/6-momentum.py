#!/usr/bin/env python3
"""Defines a function that sets up the gradient descent with
momentum optimization algorithm in tensorflow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization
    algorithm in tensorflow

    Args:
        alpha: the learning rate
        beta1: the momentum weight

    Returns:
        optimizer
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )

    return optimizer
