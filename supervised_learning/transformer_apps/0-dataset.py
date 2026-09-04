#!/usr/bin/env python3
"""
Module for loading and preparing dataset and tokenizers for machine translation.
"""

from setup import load_pt2en
import tensorflow as tf
from transformers import AutoTokenizer


class Dataset:
    """
    Dataset class that loads and preps dataset and tokenizers for machine translation.
    """

    def __init__(self):
        """
        Class constructor for Dataset.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset.

        Args:
            data (tf.data.Dataset): Dataset whose examples are (pt, en) tuples.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
        """
        def pt_corpus():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_corpus():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_corpus(), vocab_size=2**13
        )

        tokenizer_en = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_corpus(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
