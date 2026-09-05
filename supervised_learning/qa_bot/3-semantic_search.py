#!/usr/bin/env python3
"""
Module that defines a function to perform semantic search on a
corpus of reference documents using the Universal Sentence Encoder.
"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): the path to the corpus of reference
            documents on which to perform semantic search.
        sentence (str): the sentence from which to perform semantic
            search.

    Returns:
        str: the reference text of the document most similar to
            sentence.
    """
    documents = [sentence]
    filepaths = []

    for filename in os.listdir(corpus_path):
        if not filename.endswith('.md'):
            continue
        filepath = os.path.join(corpus_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            documents.append(f.read())
        filepaths.append(filepath)

    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')
    embeddings = model(documents)

    correlation = np.inner(embeddings[0], embeddings[1:])
    closest = np.argmax(correlation)

    with open(filepaths[closest], 'r', encoding='utf-8') as f:
        return f.read()
