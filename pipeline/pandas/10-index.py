#!/usr/bin/env python3
"""Module for indexing a DataFrame by Timestamp."""


def index(df):
    """Set the Timestamp column as the index of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to modify.

    Returns:
        pd.DataFrame: The DataFrame with Timestamp as index.
    """
    return df.set_index('Timestamp')
