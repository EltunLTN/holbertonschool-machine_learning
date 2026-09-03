#!/usr/bin/env python3
"""
Defines a function to create a bag-of-words embedding matrix.
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    - sentences: list of sentences to analyze
    - vocab: list of vocabulary words to use for analysis (default: None)

    Returns:
    - embeddings: numpy.ndarray of shape (s, f) containing word frequencies
    - features: list of features used for embeddings
    """
    # Preprocess sentences: convert to lowercase and strip punctuation/possessives ('s)
    processed_sentences = []
    for sentence in sentences:
        # Convert to lowercase
        text = sentence.lower()
        # Remove 's (possessive)
        text = re.sub(r"'s\b", '', text)
        # Extract word tokens matching alphanumeric sequences
        words = re.findall(r'\b\w+\b', text)
        processed_sentences.append(words)

    # Determine vocabulary / features list
    if vocab is None:
        all_words = set()
        for words in processed_sentences:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = vocab

    # Build feature-to-index mapping for fast lookup
    feat_map = {word: i for i, word in enumerate(features)}

    # Initialize the embeddings matrix (s: number of sentences, f: number of features)
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Populate sentence word counts
    for idx, words in enumerate(processed_sentences):
        for word in words:
            if word in feat_map:
                embeddings[idx, feat_map[word]] += 1

    return embeddings, features
