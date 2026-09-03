#!/usr/bin/env python3
"""
Module for calculating the cumulative n-gram BLEU score for machine translation.
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list of list of str): A list of reference translations.
        sentence (list of str): A list containing the model proposed sentence.
        n (int): The size of the largest n-gram to use for evaluation.

    Returns:
        float: The cumulative n-gram BLEU score.
    """
    if not sentence or n <= 0:
        return 0.0

    cand_len = len(sentence)

    # Find the reference length closest to the candidate sentence length.
    closest_ref = min(references, key=lambda ref: (abs(len(ref) - cand_len), len(ref)))
    ref_len = len(closest_ref)

    # Calculate Brevity Penalty (BP)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / cand_len)

    precisions = []

    for k in range(1, n + 1):
        cand_ngrams = [tuple(sentence[i:i + k]) for i in range(cand_len - k + 1)]
        total_cand_ngrams = len(cand_ngrams)

        if total_cand_ngrams == 0:
            precisions.append(0.0)
            continue

        # Count n-grams in the candidate sentence
        cand_counts = {}
        for ng in cand_ngrams:
            cand_counts[ng] = cand_counts.get(ng, 0) + 1

        # Find the maximum count of each n-gram across all reference translations
        max_ref_counts = {}
        for ng in cand_counts:
            max_count = 0
            for ref in references:
                ref_ngrams = [tuple(ref[i:i + k]) for i in range(len(ref) - k + 1)]
                ref_count = ref_ngrams.count(ng)
                if ref_count > max_count:
                    max_count = ref_count
            max_ref_counts[ng] = max_count

        # Clip counts and sum them up
        clipped_sum = 0
        for ng, count in cand_counts.items():
            clipped_sum += min(count, max_ref_counts.get(ng, 0))

        precision = clipped_sum / total_cand_ngrams
        precisions.append(precision)

    # If any precision is 0, geometric mean is 0
    if any(p == 0 for p in precisions):
        return 0.0

    # Calculate geometric mean of precisions with equal weights (1/n)
    log_sum = sum(np.log(p) for p in precisions)
    geometric_mean = np.exp(log_sum / n)

    return bp * geometric_mean
