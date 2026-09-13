#!/usr/bin/env python3
"""Defines a function that initializes all variables required to
calculate the P affinities in t-SNE."""
import numpy as np


def P_init(X, perplexity):
    """Initializes all variables required to calculate the P
    affinities in t-SNE.

    Args:
        X (numpy.ndarray): array of shape (n, d) containing the
            dataset to be transformed by t-SNE, where n is the
            number of data points and d is the number of dimensions
            in each point.
        perplexity (float): the perplexity that all Gaussian
            distributions should have.

    Returns:
        tuple: (D, P, betas, H)
            D (numpy.ndarray): array of shape (n, n) that calculates
                the squared pairwise distance between two data
                points. The diagonal of D is 0's.
            P (numpy.ndarray): array of shape (n, n) initialized to
                all 0's that will contain the P affinities.
            betas (numpy.ndarray): array of shape (n, 1) initialized
                to all 1's that will contain all of the beta values.
            H (float): the Shannon entropy for perplexity perplexity
                with a base of 2.
    """
    n, d = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
