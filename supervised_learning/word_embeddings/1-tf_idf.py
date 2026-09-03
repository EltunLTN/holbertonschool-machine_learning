#!/usr/bin/env python3
"""
Defines a function to calculate TF-IDF embedding matrix.
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Parameters:
    - sentences: list of sentences to analyze
    - vocab: list of vocabulary words to use for analysis (default: None)

    Returns:
    - embeddings: numpy.ndarray of shape (s, f) containing TF-IDF scores
    - features: list of features used for embeddings
    """
    # Preprocess sentences: convert to lowercase and extract word tokens
    processed_sentences = []
    for sentence in sentences:
        text = sentence.lower()
        text = re.sub(r"'s\b", '', text)
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

    feat_map = {word: i for i, word in enumerate(features)}
    s = len(sentences)
    f = len(features)

    # Step 1: Calculate Term Frequency (TF) for each sentence
    tf_matrix = np.zeros((s, f), dtype=float)
    for idx, words in enumerate(processed_sentences):
        word_count = len(words)
        if word_count == 0:
            continue
        # Count frequencies of each word in the sentence
        counts = {}
        for word in words:
            if word in feat_map:
                counts[word] = counts.get(word, 0) + 1
        
        # TF = (frequency of term in doc) / (total terms in doc)
        for word, count in counts.items():
            tf_matrix[idx, feat_map[word]] = count / word_count

    # Step 2: Calculate Inverse Document Frequency (IDF)
    # Using standard sklearn-style smoothing: log((1 + n_docs) / (1 + df)) + 1
    idf_vector = np.zeros(f, dtype=float)
    for j, word in enumerate(features):
        doc_count = sum(1 for words in processed_sentences if word in words)
        idf_vector[j] = np.log((1 + s) / (1 + doc_count)) + 1

    # Step 3: Compute TF-IDF and apply L2 normalization row-wise
    embeddings = tf_matrix * idf_vector
    
    # L2 normalization over rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for rows with all zeros
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings, features
