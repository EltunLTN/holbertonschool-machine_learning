#!/usr/bin/env python3
"""
Module for MultiHeadAttention class used in transformer models.
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention class for performing multi-head attention.
    """

    def __init__(self, dm, h):
        """
        Class constructor for MultiHeadAttention.

        Args:
            dm (int): dimensionality of the model.
            h (int): number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        Transpose the result such that the shape is (batch, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Forward pass for MultiHeadAttention.

        Args:
            Q (tf.Tensor): query matrix tensor of shape (batch, seq_len_q, dk).
            K (tf.Tensor): key matrix tensor of shape (batch, seq_len_v, dk).
            V (tf.Tensor): value matrix tensor of shape (batch, seq_len_v, dv).
            mask (tf.Tensor): optional mask tensor.

        Returns:
            tuple:
                output (tf.Tensor): attention output of shape
                  (..., seq_len_q, dm).
                weights (tf.Tensor): attention weights of shape
                  (..., h, seq_len_q, seq_len_v).
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm)
        )

        output = self.linear(concat_attention)

        return output, weights
