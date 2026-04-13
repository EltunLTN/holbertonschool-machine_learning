#!/usr/bin/env python3
"""Sum total"""


def summation_i_squared(n):
    """sum total"""
    if not isinstance(n, int):
        return None

    if n <= 0:
        return 0
    return n*(n+1)*(2*n+1)/6
