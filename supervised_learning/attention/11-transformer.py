#!/usr/bin/env python3
"""
Module for Transformer class used in transformer network models.
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer network class.
    """

    def __init__(
        self,
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_seq_input,
        max_seq_target,
        drop_rate=0.1
    ):
        """
        Class constructor for Transformer.

        Args:
            N (int): number of blocks in encoder and decoder.
            dm (int): dimensionality of the model.
            h (int): number of heads.
            hidden (int): number of hidden units in fully connected layers.
            input_vocab (int): size of input vocabulary.
            target_vocab (int): size of target vocabulary.
            max_seq_input (int): maximum sequence length for input.
            max_seq_target (int): maximum sequence length for target.
            drop_rate (float): dropout rate.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(
        self,
        inputs,
        target,
        training,
        encoder_mask,
        look_ahead_mask,
        decoder_mask
    ):
        """
        Forward pass for the Transformer network.

        Args:
            inputs (tf.Tensor): input tensor of shape (batch, input_seq_len).
            target (tf.Tensor): target tensor of shape (batch, target_seq_len).
            training (bool): boolean to determine if training.
            encoder_mask (tf.Tensor): padding mask for encoder.
            look_ahead_mask (tf.Tensor): look ahead mask for decoder.
            decoder_mask (tf.Tensor): padding mask for decoder.

        Returns:
            tf.Tensor: transformer output of shape
              (batch, target_seq_len, target_vocab).
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)

        decoder_output = self.decoder(
            target,
            encoder_output,
            training,
            look_ahead_mask,
            decoder_mask
        )

        output = self.linear(decoder_output)

        return output
