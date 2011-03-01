# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

cimport cython
from numpywrap cimport *

from pdfs cimport CPdf, GaussPdf


cdef class Filter:
    cpdef CPdf bayes(self, ndarray y_t)

cdef class KalmanFilter:
    cdef public ndarray A, B, C, D, Q, R
    cdef readonly int n, k, j
    cdef readonly GaussPdf P, S
    cdef bint _bayes_type_check


    @cython.locals(K = ndarray)
    cpdef ndarray bayes(self, ndarray yt, ndarray ut)
