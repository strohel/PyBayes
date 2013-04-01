# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - python version"""

from numpy import *


def vector(size):
    return empty(size)

def index_vector(size):
    return empty(size, dtype=int)

def index_range(start, stop):
    return arange(start, stop, dtype=int)

def matrix(rows, cols):
    return empty((rows, cols))

def reindex_mv(data, indices):
    data[:] = data[indices]
reindex_vv = reindex_mv

# NumPy doesn't differentiate between vectors and matrices, Ceygen does:
add_vv = add
add_mm = add
dot_vv = dot
dot_vm = dot
dot_mv = dot
dot_mm = dot
multiply_vs = multiply
power_vs = power
put_vv = put
subtract_vv = subtract
subtract_mm = subtract
sum_v = sum
take_vv = take
