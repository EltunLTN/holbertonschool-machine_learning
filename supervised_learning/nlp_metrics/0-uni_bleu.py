#!/usr/bin/env python3
"""
Module for calculating the unigram BLEU score for machine translation.
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list of list of str): A list of reference translations.
        sentence (list of str): A list containing the model proposed sentence.

    Returns:
        float: The unigram BLEU score.
    """
    if not sentence:
        return 0.0

    cand_len = len(sentence)

    # Find the reference length closest to the candidate sentence length.
    # In case of a tie in absolute difference, choose the shorter reference length.
    closest_ref = min(references, key=lambda ref: (abs(len(ref) - cand_len), len(ref)))
    ref_len = len(closest_ref)

    # Calculate Brevity Penalty (BP)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / cand_len)

    # Count unigrams in the candidate sentence
    cand_counts = {}
    for word in sentence:
        cand_counts[word] = cand_counts.get(word, 0) + 1

    # Find the maximum count of each unigram across all reference translations
    max_ref_counts = {}
    for word in cand_counts:
        max_count = 0
        for ref in references:
            ref_count = ref.count(word)
            if ref_count > max_count:
                max_count = ref_count
        max_ref_counts[word] = max_count

    # Clip counts and sum them up
    clipped_sum = 0
    for word, count in cand_counts.items():
        clipped_sum += min(count, max_ref_counts.get(word, 0))

    # Calculate unigram precision
    precision = clipped_sum / cand_len

    return bp * precision
