#!/usr/bin/env python3
"""Module that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Returns the integral of a polynomial"""

    # Validation
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, int):
        return None

    for item in poly:
        if not isinstance(item, (int, float)):
            return None

    # Integral hesabla
    result = [C]

    for i in range(len(poly)):
        val = poly[i] / (i + 1)

        # Əgər tam ədəddirsə → int et
        if val == int(val):
            val = int(val)

        result.append(val)

    # Listi "minimal" et (sondakı 0-ları sil)
    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
