#!/usr/bin/env python3
"""
Neural Style Transfer.
"""
import numpy as np
import tensorflow as tf


class NST:
    """Represents a neural style transfer model."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize an NST instance."""
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[-1] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[-1] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)'
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    def load_model(self):
        """Load VGG19 and build the feature-extraction model."""
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False
        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers + [self.content_layer]
        ]
        self.model = tf.keras.Model(vgg.input, outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a rank-4 tensor."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                input_layer.shape.rank != 4):
            raise TypeError('input_layer must be a tensor of rank 4')
        gram = tf.matmul(
            tf.transpose(
                tf.reshape(
                    input_layer,
                    (tf.shape(input_layer)[0], -1, tf.shape(input_layer)[-1])
                ),
                perm=[0, 2, 1]
            ),
            tf.reshape(
                input_layer,
                (tf.shape(input_layer)[0], -1, tf.shape(input_layer)[-1])
            )
        )
        return gram / tf.cast(
            tf.shape(input_layer)[1] * tf.shape(input_layer)[2],
            tf.float32
        )
