#!/usr/bin/env python3
"""Module for concatenating two DataFrames."""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate bitstamp and coinbase DataFrames.

    Args:
        df1 (pd.DataFrame): The coinbase DataFrame.
        df2 (pd.DataFrame): The bitstamp DataFrame.

    Returns:
        pd.DataFrame: The concatenated DataFrame with keys.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index <= 1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
