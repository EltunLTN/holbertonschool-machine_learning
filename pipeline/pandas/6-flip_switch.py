#!/usr/bin/env python3
"""
Defines the flip_switch function for sorting and transposing a DataFrame.
"""
import pandas as pd


def flip_switch(df):
    """
    Sorts a DataFrame in reverse chronological order and transposes it.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Sort the index in descending order to get reverse chronological order,
    # then transpose using .T (or .transpose())
    return df.sort_index(ascending=False).T
