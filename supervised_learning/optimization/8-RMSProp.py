#!/usr/bin/env python3
"""Defines a function that sets up the RMSProp optimization
algorithm in tensorflow"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in tensorflow

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight (Discounting factor)
        epsilon: a small number to avoid division by zero

    Returns:
        optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
