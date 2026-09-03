#!/usr/bin/env python3

import tensorflow.keras as K


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model into a trainable Keras Embedding layer.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        Keras Embedding layer initialized with the Word2Vec weights
    """
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape

    embedding = K.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embedding
