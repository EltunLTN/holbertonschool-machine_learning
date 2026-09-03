#!/usr/bin/env python3
"""
LSTM Cell Module
"""

import numpy as np


class LSTMCell:
    """Represents a long short-term memory unit cell."""

    def __init__(self, i, h, o):
        """Class constructor."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step."""
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))

        # Update gate
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))

        # Intermediate cell state
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_tilde

        # Output gate
        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y
