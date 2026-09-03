#!/usr/bin/env python3
"""
GRU Cell Module
"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """Class constructor."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))

        # Reset gate
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        # Candidate hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
