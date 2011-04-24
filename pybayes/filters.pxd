# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

cimport cython
from numpywrap cimport *

from pdfs cimport CPdf, Pdf, GaussPdf, EmpPdf


cdef class Filter(object):
    cpdef bint bayes(self, ndarray yt, ndarray ut = *) except False
    cpdef Pdf posterior(self)
    cpdef double evidence_log(self, ndarray yt) except? -1


cdef class KalmanFilter(Filter):
    cdef readonly ndarray A, B, C, D, Q, R
    cdef readonly int n, k, j
    cdef readonly GaussPdf P, S

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __copy__(self)

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __deepcopy__(self, memo)

    @cython.locals(K = ndarray)
    cpdef bint bayes(self, ndarray yt, ndarray ut = *) except False


cdef class ParticleFilter(Filter):
    cdef readonly CPdf p_xt_xtp, p_yt_xt
    cdef readonly EmpPdf emp
