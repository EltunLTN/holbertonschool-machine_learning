#!/usr/bin/env python3
"""Perform a same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Perform a same convolution on a batch of grayscale images."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )
    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved
