#!/usr/bin/env python3
"""
Module for RNNDecoder class used in
sequence-to-sequence models with attention.
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class for machine
    translation decoding with attention.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor for RNNDecoder.

        Args:
            vocab (int): size of the output vocabulary.
            embedding (int): dimensionality of the embedding vector.
            units (int): number of hidden units in the RNN cell.
            batch (int): batch size.
        """
        super(RNNDecoder, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding
        self.units = units
        self.batch = batch

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units=units)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass for the decoder layer.

        Args:
            x (tf.Tensor): tensor of shape (batch, 1)
              containing previous word index.
            s_prev (tf.Tensor): tensor of shape (batch, units)
              containing previous hidden state.
            hidden_states (tf.Tensor): tensor of shape
              (batch, input_seq_len, units) containing encoder outputs.

        Returns:
            tuple:
                y (tf.Tensor): output word scores of shape (batch, vocab).
                s (tf.Tensor): new decoder hidden state of shape (batch, units).
        """
        x = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        context_expand = tf.expand_dims(context, 1)
        x = tf.concat([context_expand, x], axis=-1)

        outputs, s = self.gru(x, initial_state=s_prev)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
