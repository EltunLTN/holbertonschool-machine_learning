#!/usr/bin/env python3
"""
Module for Encoder class used in transformer models.
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class for creating the encoder for a transformer.
    """

    def __init__(
        self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1
    ):
        """
        Class constructor for Encoder.

        Args:
            N (int): number of blocks in the encoder.
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units
              in the fully connected layer.
            input_vocab (int): size of the input vocabulary.
            max_seq_len (int): maximum sequence length possible.
            drop_rate (float): dropout rate.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab, output_dim=dm
        )
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the Encoder.

        Args:
            x (tf.Tensor): input tensor of shape
              (batch, input_seq_len).
            training (bool): boolean to determine
              if the model is training.
            mask (tf.Tensor): mask to be applied
              for multi head attention.

        Returns:
            tf.Tensor: encoder output tensor of shape
              (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x
