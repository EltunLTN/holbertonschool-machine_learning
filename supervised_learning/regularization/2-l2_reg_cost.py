#!/usr/bin/env python3
"""
Calculate the cost of a neural network with L2 regularization using a Keras model.
"""

import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network accounting for L2 regularization.

    Parameters:
    - cost: tensor, cost of the network without L2 regularization
    - model: Keras model with layers that include L2 regularization

    Returns:
    - tensor, total cost including L2 regularization for each layer
    """
    if model.losses:
        l2_losses = tf.add_n(model.losses)
        return cost + l2_losses
    return cost
