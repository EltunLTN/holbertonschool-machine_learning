#!/usr/bin/env python3


def poly_derivative(poly):
    new_list = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for i in range(len(poly)):
        new_list.append(poly[i]*(i))
    del new_list[0]
    return new_list
