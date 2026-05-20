#!/usr/bin/env python3
"""Module for renaming and converting DataFrame columns."""
import pandas as pd


def rename(df):
    """Rename Timestamp column to Datetime and convert values to datetime.

    Args:
        df: pd.DataFrame containing a Timestamp column.

    Returns:
        Modified pd.DataFrame with only Datetime and Close columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
