#!/usr/bin/env python3
"""Module for analyzing descriptive statistics of a DataFrame."""


def analyze(df):
    """Compute descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.

    Returns:
        pd.DataFrame: Descriptive statistics for all non-Timestamp columns.
    """
    return df.drop(columns=['Timestamp']).describe()
