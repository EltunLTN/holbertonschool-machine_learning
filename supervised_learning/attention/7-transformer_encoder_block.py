#!/usr/bin/env python3
"""
Module for EncoderBlock class used in transformer models.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    EncoderBlock class for creating an
    encoder block for a transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor for EncoderBlock.

        Args:
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units
              in the fully connected layer.
            drop_rate (float): dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass for the EncoderBlock.

        Args:
            x (tf.Tensor): input tensor of shape
              (batch, input_seq_len, dm).
            training (bool): boolean to determine
              if the model is training.
            mask (tf.Tensor, optional): mask to be
              applied for multi-head attention. Defaults to None.

        Returns:
            tf.Tensor: block's output tensor of shape
              (batch, input_seq_len, dm).
        """
        # Multi-head attention block
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward block
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
