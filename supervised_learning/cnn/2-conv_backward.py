#!/usr/bin/env python3
"""Module that performs back propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Perform back propagation over a convolutional layer of a NN.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated
            output of the convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
            kh is the filter height
            kw is the filter width
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the
            biases applied to the convolution
        padding: string that is either 'same' or 'valid', indicating
            the type of padding used
        stride: tuple of (sh, sw) containing the strides for the
            convolution
            sh is the stride for the height
            sw is the stride for the width

    Returns:
        The partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_new, w_new, c_new = dZ.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    dA_prev_padded = np.zeros(A_prev_padded.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    db[0, 0, 0, :] = np.sum(dZ, axis=(0, 1, 2))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                A_slice = A_prev_padded[
                    :, vert_start:vert_end, horiz_start:horiz_end, :
                ]

                dW[:, :, :, k] += np.sum(
                    A_slice * dZ[:, i, j, k].reshape(-1, 1, 1, 1),
                    axis=0
                )

                dA_prev_padded[
                    :, vert_start:vert_end, horiz_start:horiz_end, :
                ] += W[:, :, :, k] * dZ[:, i, j, k].reshape(-1, 1, 1, 1)

    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
