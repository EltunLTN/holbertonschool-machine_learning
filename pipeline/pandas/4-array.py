#!/usr/bin/env python3
"""Module for converting DataFrame columns to numpy array."""


def array(df):
    """Select last 10 rows of High and Close columns as a numpy.ndarray.

    Args:
        df: pd.DataFrame containing High and Close columns.

    Returns:
        numpy.ndarray of the last 10 rows of High and Close columns.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
