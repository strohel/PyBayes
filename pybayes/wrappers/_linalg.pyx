# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy.linalg - cython version"""

from ceygen.lu cimport inv as c_inv
from numpy cimport import_array, PyArray_EMPTY, NPY_DOUBLE
from numpy.linalg import *


import_array()  # needed to call any PyArray functions

cdef ndarray inv(ndarray A):
    cdef ndarray R = PyArray_EMPTY(2, A.shape, NPY_DOUBLE, 0)
    c_inv(A, R)
    return R
