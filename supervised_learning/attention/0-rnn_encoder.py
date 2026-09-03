#!/usr/bin/env python3
"""
Module for RNNEncoder class used in
sequence-to-sequence models with attention.
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder class for machine translation.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor for RNNEncoder.

        Args:
            vocab (int): size of the input vocabulary.
            embedding (int): dimensionality of the embedding vector.
            units (int): number of hidden units in the RNN cell.
            batch (int): batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN
        cell to a tensor of zeros.

        Returns:
            tf.Tensor: A tensor of shape (batch, units) with zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Forward pass for the encoder layer.

        Args:
            x (tf.Tensor): Input tensor of shape
              (batch, input_seq_len).
            initial (tf.Tensor): Initial hidden state
              of shape (batch, units).

        Returns:
            tuple:
                outputs (tf.Tensor): Outputs of shape
                  (batch, input_seq_len, units).
                hidden (tf.Tensor): Last hidden state
                  of shape (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
