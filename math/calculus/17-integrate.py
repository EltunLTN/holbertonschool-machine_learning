#!/usr/bin/env python3
"""Module contains the function poly_integral
that calculates the integral of a polynomial"""


def poly_integral(poly):
    """Calculates the integral of a polynomial"""
    new_list = [0]
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for i in range(len(poly)):
        new_list.append(poly[i]/(i+1))

    if len(new_list) == 0:
        return [0]
    return new_list
