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
        self.generate_features()

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

    def generate_features(self):
        """Extract target style Gram matrices and content features."""
        vgg19 = tf.keras.applications.vgg19
        style_input = vgg19.preprocess_input(self.style_image * 255)
        content_input = vgg19.preprocess_input(self.content_image * 255)
        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)
        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculate style cost for one layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                style_output.shape.rank != 4):
            raise TypeError('style_output must be a tensor of rank 4')
        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape.rank != 3 or
                gram_target.shape[0] != 1 or
                gram_target.shape[1] != c or gram_target.shape[2] != c):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}] '
                'where {} is the number of channels in style_output'.format(
                    c, c, c)
            )
        gram_style = self.gram_matrix(style_output)
        return tf.reduce_sum(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculate the total style cost."""
        length = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != length:
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    length
                )
            )
        weights = 1.0 / length
        return weights * tf.add_n([
            self.layer_style_cost(output, target)
            for output, target in zip(style_outputs, self.gram_style_features)
        ])

    def content_cost(self, content_output):
        """Calculate content cost."""
        shape = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != shape):
            raise TypeError(
                'content_output must be a tensor of shape {}'.format(shape)
            )
        return tf.reduce_sum(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """Calculate total, content, and style costs."""
        shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(shape)
            )
        vgg19 = tf.keras.applications.vgg19
        processed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(processed)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]
        j_content = self.content_cost(content_output)
        j_style = self.style_cost(style_outputs)
        j_total = self.alpha * j_content + self.beta * j_style
        return j_total, j_content, j_style

    def compute_grads(self, generated_image):
        """Calculate image gradients and all costs."""
        shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(shape)
            )
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            j_total, j_content, j_style = self.total_cost(generated_image)
        grads = tape.gradient(j_total, generated_image)
        return grads, j_total, j_content, j_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generate a style-transferred image using Adam."""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be positive')
        if step is not None and not isinstance(step, int):
            raise TypeError('step must be an integer')
        if step is not None and (step <= 0 or step > iterations):
            raise ValueError(
                'step must be positive and less than iterations'
            )
        if not isinstance(lr, (int, float)):
            raise TypeError('lr must be a number')
        if lr <= 0:
            raise ValueError('lr must be positive')
        if not isinstance(beta1, float):
            raise TypeError('beta1 must be a float')
        if beta1 < 0 or beta1 > 1:
            raise ValueError('beta1 must be in the range [0, 1]')
        if not isinstance(beta2, float):
            raise TypeError('beta2 must be a float')
        if beta2 < 0 or beta2 > 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        generated_image = tf.Variable(self.content_image)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )
        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, j_total, j_content, j_style = self.compute_grads(
                generated_image
            )
            optimizer.apply_gradients([(grads, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

            if j_total < best_cost:
                best_cost = j_total
                best_image = tf.identity(generated_image)

            if step is not None and (i % step == 0 or i == iterations):
                print(
                    'Cost at iteration {}: {}, content {}, style {}'.format(
                        i, j_total, j_content, j_style
                    )
                )
        return best_image, best_cost
