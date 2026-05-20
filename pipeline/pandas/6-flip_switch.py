#!/usr/bin/env python3
"""Module for flipping and switching a DataFrame."""


def flip_switch(df):
    """Sort DataFrame in reverse chronological order and transpose it.

    Args:
        df (pd.DataFrame): The input DataFrame to transform.

    Returns:
        pd.DataFrame: The sorted and transposed DataFrame.
    """
    return df.sort_values(by='Timestamp', ascending=False).transpose()
