#!/usr/bin/env python3
"""
Module for creating attention and padding masks for the Transformer model.
"""

import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a given sequence.

    Args:
        seq (tf.Tensor): Sequence tensor.

    Returns:
        tf.Tensor: Padding mask tensor.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for future tokens.

    Args:
        size (int): Size of the look-ahead mask.

    Returns:
        tf.Tensor: Look-ahead mask tensor.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs (tf.Tensor): Input sentence tensor of shape (batch_size, seq_len_in).
        target (tf.Tensor): Target sentence tensor of shape (batch_size, seq_len_out).

    Returns:
        tuple: (encoder_mask, combined_mask, decoder_mask)
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
