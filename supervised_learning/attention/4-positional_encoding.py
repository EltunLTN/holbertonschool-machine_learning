#!/usr/bin/env python3
"""
Module for calculating positional encoding for a transformer.
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.

    Args:
        max_seq_len (int): maximum sequence length.
        dm (int): model depth.

    Returns:
        numpy.ndarray: positional encoding vectors of shape (max_seq_len, dm).
    """
    pos = np.arange(max_seq_len, dtype=np.float32)
    i = np.arange(dm, dtype=np.float32)

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / dm)
    angle_rads = pos[:, np.newaxis] * angle_rates[np.newaxis, :]

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads
