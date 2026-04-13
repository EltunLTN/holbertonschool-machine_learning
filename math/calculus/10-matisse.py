#!/usr/bin/env python3
"""Module contains the function poly_derivative
that calculates the derivative of a polynomial"""

def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    new_list = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for i in range(len(poly)):
        new_list.append(poly[i]*(i))
    del new_list[0]
    if len(new_list) == 0:
        return [0]
    return new_list
