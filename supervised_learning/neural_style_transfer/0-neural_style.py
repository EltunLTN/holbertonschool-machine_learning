#!/usr/bin/env python3
"""Neural Style Transfer module."""

import tensorflow as tf
import numpy as np


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST instance.

        Args:
            style_image: Style reference image.
            content_image: Content reference image.
            alpha: Weight for content cost.
            beta: Weight for style cost.

        Raises:
            TypeError: If an argument has an invalid type or value.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )

        if (not isinstance(alpha, (int, float))
                or alpha < 0):
            raise TypeError(
                'alpha must be a non-negative number'
            )

        if (not isinstance(beta, (int, float))
                or beta < 0):
            raise TypeError(
                'beta must be a non-negative number'
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Scale an image so its largest side is 512 pixels.

        Args:
            image: Image as a numpy.ndarray of shape (h, w, 3).

        Returns:
            A TensorFlow tensor of shape (1, h_new, w_new, 3).

        Raises:
            TypeError: If image is not a valid numpy.ndarray.
        """
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)'
            )

        image = tf.convert_to_tensor(image, dtype=tf.float32)

        height = image.shape[0]
        width = image.shape[1]

        if height >= width:
            new_height = 512
            new_width = round(width * 512 / height)
        else:
            new_width = 512
            new_height = round(height * 512 / width)

        image = tf.image.resize(
            image,
            (new_height, new_width),
            method='bicubic'
        )

        image = image / 255.0

        return tf.expand_dims(image, axis=0)
