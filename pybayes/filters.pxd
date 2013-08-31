# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

cimport cython

cimport pybayes.wrappers._linalg as linalg
cimport pybayes.wrappers._numpy as np

from pdfs cimport CPdf, Pdf, GaussPdf, EmpPdf, MarginalizedEmpPdf


# workarounds for Cython:
ctypedef double[:, :] double_2D
ctypedef int[:] int_1D

cdef class Filter(object):
    cpdef bint bayes(self, double[:] yt, double[:] cond = *) except False
    cpdef CPdf posterior(self)
    cpdef double evidence_log(self, double[:] yt) except? -1


cdef class KalmanFilter(Filter):
    cdef readonly double[:, :] A, B, C, D, Q, R
    cdef readonly int n, k, j
    cdef readonly GaussPdf P, S

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __copy__(self)

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __deepcopy__(self, memo)

    @cython.locals(K = double_2D)
    cpdef bint bayes(self, double[:] yt, double[:] cond = *) except False


cdef class ParticleFilter(Filter):
    cdef readonly CPdf p_xt_xtp, p_yt_xt
    cdef readonly EmpPdf emp
    cdef readonly Filter proposal

    cpdef bint bayes(self, double[:] yt, double[:] cond = *) except False


cdef class MarginalizedParticleFilter(Filter):
    cdef readonly CPdf p_bt_btp
    cdef readonly KalmanFilter[:] kalmans
    cdef readonly MarginalizedEmpPdf memp

    @cython.locals(kalman = KalmanFilter)
    cpdef bint bayes(self, double[:] yt, double[:] cond = *) except False

    @cython.locals(indices = int_1D)
    cpdef bint _resample(self) except False
