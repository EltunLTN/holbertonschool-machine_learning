#!/usr/bin/env python3
"""
RNN Forward Propagation Module
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN."""
    t, m, i = X.shape
    _, h = h_0.shape
    _, o = rnn_cell.Wy.shape

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0
    h_prev = h_0

    for step in range(t):
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y
        h_prev = h_next

    return H, Y
