#!/usr/bin/env python3
"""Perform max or average pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Pool a batch of multi-channel images using max or average values."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1
    pooled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            row = i * sh
            column = j * sw
            region = images[:, row:row + kh, column:column + kw, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(region, axis=(1, 2))

    return pooled
