#!/usr/bin/env python3
"""
Module for DecoderBlock class used in transformer models.
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    DecoderBlock class for creating a
    decoder block for a transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor for DecoderBlock.

        Args:
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units
              in the fully connected layer.
            drop_rate (float): dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
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
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the DecoderBlock.

        Args:
            x (tf.Tensor): input tensor of shape
              (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): encoder output tensor
              of shape (batch, input_seq_len, dm).
            training (bool): boolean to determine
              if the model is training.
            look_ahead_mask (tf.Tensor): mask to be applied
              to the first multi-head attention layer.
            padding_mask (tf.Tensor): mask to be applied
              to the second multi-head attention layer.

        Returns:
            tf.Tensor: block's output tensor of shape
              (batch, target_seq_len, dm).
        """
        # 1. Masked multi-head self-attention
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 2. Encoder-decoder multi-head attention
        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # 3. Feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
