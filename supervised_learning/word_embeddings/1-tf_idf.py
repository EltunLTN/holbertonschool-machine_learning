#!/usr/bin/env python3
"""
Creates a TF-IDF embedding matrix
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.

    Args:
        sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis.
               If None, all words within sentences should be used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings
        features: numpy.ndarray of the features used for embeddings
    """
    # Initialize the vectorizer with the given vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    
    # Fit and transform the sentences into TF-IDF embeddings
    X = vectorizer.fit_transform(sentences)
    
    # Convert sparse matrix to dense numpy array
    embeddings = X.toarray()
    
    # Extract feature names (handling different sklearn versions)
    if hasattr(vectorizer, 'get_feature_names_out'):
        features = vectorizer.get_feature_names_out()
    else:
        features = vectorizer.get_feature_names()
        
    # Ensure features is a numpy array to match the expected checker output format
    features = np.array(features)
    
    return embeddings, features
