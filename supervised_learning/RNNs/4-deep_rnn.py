#!/usr/bin/env python3
"""
Deep RNN Forward Propagation Module
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN."""
    l = len(rnn_cells)
    t, m, i = X.shape
    _, _, h = h_0.shape
    _, o = rnn_cells[-1].Wy.shape

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            h_prev = H[step, layer]
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            else:
                h_next, y = rnn_cells[layer].forward(h_prev, current_x)
            
            current_x = h_next
            H[step + 1, layer] = h_next
        
        Y[step] = y

    return H, Y
