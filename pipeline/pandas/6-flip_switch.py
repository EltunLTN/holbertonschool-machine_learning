#!/usr/bin/env python3
"""
Module that flips and switches a DataFrame.
"""

import pandas as pd


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order
    and transposes it.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    return df.sort_index(ascending=False).transpose()
