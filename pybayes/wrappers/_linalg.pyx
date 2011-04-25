# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy.linalg - cython version"""

from numpy cimport import_array, PyArray_EMPTY, NPY_INT32
from numpy.linalg import *

cimport tokyo as t


import_array()  # needed to call any PyArray functions

cdef ndarray inv(ndarray A):
    cdef ndarray p = PyArray_EMPTY(1, A.shape, NPY_INT32, 0)
    cdef ndarray R = A.copy()  # TODO: optimisation possible

    if t.dgetrf(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetrf)")
    if t.dgetri(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetri)")
    return R
