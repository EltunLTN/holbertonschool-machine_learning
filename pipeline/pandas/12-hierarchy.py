#!/usr/bin/env python3
"""Module for creating a hierarchical concatenation of two DataFrames."""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenate bitstamp and coinbase with Timestamp as top level.

    Args:
        df1 (pd.DataFrame): The coinbase DataFrame.
        df2 (pd.DataFrame): The bitstamp DataFrame.

    Returns:
        pd.DataFrame: The concatenated DataFrame with Timestamp as top index.
    """
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1).sort_index()
    return df
