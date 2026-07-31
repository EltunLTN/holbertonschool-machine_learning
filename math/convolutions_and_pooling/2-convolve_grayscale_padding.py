#!/usr/bin/env python3
"""Perform a convolution with custom padding on grayscale images."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolve grayscale images using the specified zero padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )
    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
