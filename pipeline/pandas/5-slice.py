#!/usr/bin/env python3
"""Module for slicing a DataFrame."""


def slice(df):
    """Extract specific columns and select every 60th row.

    Args:
        df: pd.DataFrame containing High, Low, Close, and Volume_(BTC) columns.

    Returns:
        Sliced pd.DataFrame with every 60th row of the selected columns.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
