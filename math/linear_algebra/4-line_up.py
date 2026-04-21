#!/usr/bin/env python3
"""Adding array function."""


def add_arrays(arr1, arr2):
    """Adding array function."""
    if len(arr1) != len(arr2):
        return None
    list = []
    for i in range(len(arr1)):
        list.append(arr1[i] + arr2[i])
    return list
