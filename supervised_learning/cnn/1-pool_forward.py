#!/usr/bin/env python3
"""Module that performs forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform forward propagation over a pooling layer of a NN.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the
            kernel for the pooling
            kh is the kernel height
            kw is the kernel width
        stride: tuple of (sh, sw) containing the strides for the
            pooling
            sh is the stride for the height
            sw is the stride for the width
        mode: string containing either 'max' or 'avg', indicating
            whether to perform maximum or average pooling,
            respectively

    Returns:
        The output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(A_slice, axis=(1, 2))

    return A
