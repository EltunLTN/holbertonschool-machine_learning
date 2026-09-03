#!/usr/bin/env python3
"""
Converts a Gensim word2vec model to a Keras Embedding layer
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim word2vec model to a keras Embedding layer.
    
    Args:
        model: a trained gensim word2vec model
        
    Returns:
        The trainable keras Embedding
    """
    # Extract the embedding weights from the gensim model
    weights = model.wv.vectors
    
    # Extract dimensions
    vocab_size, vector_size = weights.shape
    
    # Create the Keras Embedding layer
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )
    
    return embedding
