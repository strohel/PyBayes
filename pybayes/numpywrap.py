# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various utility definitions and functions used throught PyBayes"""

import numpy as np


def dot(a, b):
    if a is None:
        raise TypeError("a must be numpy.ndarray")
    if b is None:
        raise TypeError("b must be numpy.ndarray")
