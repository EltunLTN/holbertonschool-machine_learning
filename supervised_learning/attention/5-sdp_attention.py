#!/usr/bin/env python3
"""
Module for Scaled Dot Product Attention mechanism.
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q (tf.Tensor): query matrix of shape (..., seq_len_q, dk).
        K (tf.Tensor): key matrix of shape (..., seq_len_v, dk).
        V (tf.Tensor): value matrix of shape (..., seq_len_v, dv).
        mask (tf.Tensor, optional): optional mask tensor.
          Defaults to None.

    Returns:
        tuple:
            output (tf.Tensor): scaled dot product
              attention of shape (..., seq_len_q, dv).
            weights (tf.Tensor): attention weights
              of shape (..., seq_len_q, seq_len_v).
    """
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)

    # Calculate Q * K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale by the square root of dk
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax along the last dimension to get attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply weights by V to get the output
    output = tf.matmul(weights, V)

    return output, weights
