#!/usr/bin/env python3
"""
Module for SelfAttention class used in
sequence-to-sequence models with attention.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention class for machine
    translation attention mechanism.
    """

    def __init__(self, units):
        """
        Class constructor for SelfAttention.

        Args:
            units (int): number of hidden units
            in the alignment model.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Forward pass for the attention layer.

        Args:
            s_prev (tf.Tensor): previous decoder
              hidden state of shape (batch, units).
            hidden_states (tf.Tensor): encoder
              outputs of shape (batch, input_seq_len, units).

        Returns:
            tuple:
                context (tf.Tensor): context vector
                  of shape (batch, units).
                weights (tf.Tensor): attention weights
                  of shape (batch, input_seq_len, 1).
        """
        # Expand s_prev to shape (batch, 1, units)
        # for broadcasting
        s_prev_expand = tf.expand_dims(s_prev, 1)

        # Apply dense layers W and U
        w_s_prev = self.W(s_prev_expand)
        u_hidden = self.U(hidden_states)

        # Calculate score using tanh and V dense layer
        score = self.V(tf.nn.tanh(w_s_prev + u_hidden))

        # Calculate attention weights using softmax
        # over sequence length dimension
        weights = tf.nn.softmax(score, axis=1)

        # Calculate context vector by weighted sum
        # of encoder hidden states
        context = tf.reduce_sum(
            weights * hidden_states, axis=1
        )

        return context, weights
