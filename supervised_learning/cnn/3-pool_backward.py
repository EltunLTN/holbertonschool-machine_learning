#!/usr/bin/env python3
"""Module that performs back propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Perform back propagation over a pooling layer of a NN.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output of the
            pooling layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
            containing the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
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
        The partial derivatives with respect to the previous layer
        (dA_prev).
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for n in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    vert_start = i * sh
                    vert_end = vert_start + kh
                    horiz_start = j * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        A_slice = A_prev[
                            n, vert_start:vert_end, horiz_start:horiz_end, k
                        ]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[
                            n, vert_start:vert_end, horiz_start:horiz_end, k
                        ] += mask * dA[n, i, j, k]
                    else:
                        avg_dA = dA[n, i, j, k] / (kh * kw)
                        dA_prev[
                            n, vert_start:vert_end, horiz_start:horiz_end, k
                        ] += np.ones((kh, kw)) * avg_dA

    return dA_prev
