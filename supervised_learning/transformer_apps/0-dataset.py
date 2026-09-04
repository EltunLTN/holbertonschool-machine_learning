#!/usr/bin/env python3
"""
Module for loading and preparing a dataset for machine translation.
"""
from setup import load_pt2en
import transformers


class Dataset:
    """
    Dataset class that loads and preps a dataset for machine
    translation.
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
            data (tf.data.Dataset): Dataset whose examples are
                (pt, en) tuples.
                pt (tf.Tensor): Portuguese sentence.
                en (tf.Tensor): Corresponding English sentence.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
                tokenizer_pt is the Portuguese tokenizer.
                tokenizer_en is the English tokenizer.
        """
        def pt_corpus():
            """Generator yielding decoded Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_corpus():
            """Generator yielding decoded English sentences."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_corpus(), vocab_size=2 ** 13
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_corpus(), vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
