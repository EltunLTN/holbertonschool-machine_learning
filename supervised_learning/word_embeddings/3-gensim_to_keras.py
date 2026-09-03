#!/usr/bin/env python3
"""
Defines a function to convert a Gensim Word2Vec model to a Keras Embedding layer.
"""
import tensorflow.keras as keras


def gensim_to_keras(model):
    """
    Converts a trained gensim word2vec model to a keras Embedding layer.

    Parameters:
    - model: a trained gensim Word2Vec model

    Returns:
    - keras.layers.Embedding: the trainable keras Embedding layer initialized
      with the model's vectors
    """
    # Extract the keyed vectors from the word2vec model
    wv = model.wv
    
    # Get the embedding weights matrix and dimensions
    weights = wv.vectors  # shape: (vocab_size, vector_size)
    vocab_size, vector_size = weights.shape

    # Create the Keras Embedding layer using the weights
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
