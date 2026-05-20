#!/usr/bin/env python3
"""Module for sorting a DataFrame by High price."""


def high(df):
    """Sort DataFrame by High price in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame to sort.

    Returns:
        pd.DataFrame: The DataFrame sorted by High price descending.
    """
    return df.sort_values(by='High', ascending=False)
