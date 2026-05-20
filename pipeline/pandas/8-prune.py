#!/usr/bin/env python3
"""Module for pruning NaN values from a DataFrame."""


def prune(df):
    """Remove entries where Close column has NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame to prune.

    Returns:
        pd.DataFrame: The DataFrame with NaN Close rows removed.
    """
    return df.dropna(subset=['Close'])
