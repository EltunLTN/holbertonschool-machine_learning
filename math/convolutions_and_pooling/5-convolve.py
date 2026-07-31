#!/usr/bin/env python3
"""Perform convolution on images using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Convolve a batch of multi-channel images with multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )
    output_h = ((h + (2 * ph) - kh) // sh) + 1
    output_w = ((w + (2 * pw) - kw) // sw) + 1
    convolved = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            row = i * sh
            column = j * sw
            region = padded[:, row:row + kh, column:column + kw, :]

            for kernel_index in range(nc):
                convolved[:, i, j, kernel_index] = np.sum(
                    region * kernels[:, :, :, kernel_index],
                    axis=(1, 2, 3)
                )

    return convolved
