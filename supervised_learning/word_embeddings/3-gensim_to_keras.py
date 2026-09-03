#!/usr/bin/env python3
"""Module that converts a gensim Word2Vec model to a Keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model into a trainable Keras Embedding layer

    Args:
        model: a trained gensim Word2Vec model

    Returns:
        the trainable Keras Embedding layer
    """
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        trainable=True
    )
    embedding_layer.build((None,))
    embedding_layer.set_weights([weights])

    return embedding_layer
