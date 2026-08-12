#!/usr/bin/env python3
"""Module that performs PCA color augmentation"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Args:
        image: a 3D tf.Tensor containing the image to change
        alphas: a tuple of length 3 containing the amount that each
            channel should change

    Returns:
        the augmented image
    """
    image_arr = image.numpy().astype(np.float32) / 255.0
    orig_shape = image_arr.shape

    # Reshape to a list of pixels (N, 3)
    pixels = image_arr.reshape(-1, 3)

    # Compute mean and std per channel, and center/normalize
    pixel_mean = np.mean(pixels, axis=0)
    pixel_std = np.std(pixels, axis=0)
    normalized_pixels = (pixels - pixel_mean) / pixel_std

    # Compute covariance matrix of normalized pixels
    covariance_matrix = np.cov(normalized_pixels, rowvar=False)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)

    # Build the perturbation vector
    perturbation = eig_vecs.dot(alphas * eig_vals)

    # Add perturbation (scaled back to pixel value range) to each pixel
    add_vec = perturbation * 255.0
    image_arr = image_arr * 255.0
    augmented = image_arr.reshape(orig_shape) + add_vec

    augmented = np.clip(augmented, 0, 255).astype(np.uint8)

    return tf.constant(augmented)
