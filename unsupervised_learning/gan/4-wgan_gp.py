#!/usr/bin/env python3
"""Module that defines WGAN_GP class with pre-trained weight replacement."""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """Wasserstein GAN with Gradient Penalty implementation."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """Initializes the WGAN_GP model."""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9
        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Generator loss and optimizer
        self.generator.loss = lambda fake_pred: -tf.reduce_mean(fake_pred)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss
        )

        # Discriminator loss and optimizer
        self.discriminator.loss = lambda real_pred, fake_pred: (
            tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss
        )

    def replace_weights(self, gen_h5, disc_h5):
        """Replaces generator and discriminator weights with pre-trained ones.

        Args:
            gen_h5 (str): Path to the generator weights file (.h5).
            disc_h5 (str): Path to the discriminator weights file (.h5).
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)

    def get_fake_sample(self, size=None, training=False):
        """Generates a batch of fake samples."""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Retrieves a random batch of real samples."""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Computes interpolated samples between real and fake data."""
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Calculates the gradient penalty for WGAN-GP."""
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """Executes one training step for discriminator and generator."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                interpolated_sample = self.get_interpolated_sample(
                    real_sample, fake_sample
                )

                real_pred = self.discriminator(real_sample, training=True)
                fake_pred = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(real_pred, fake_pred)
                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            disc_gradients = disc_tape.gradient(
                new_discr_loss,
                self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradients, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_sample, training=True)
            gen_loss = self.generator.loss(fake_pred)

        gen_gradients = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss,
            "gp": gp
        }
