# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

cimport cython

cimport pybayes.wrappers._linalg as linalg
cimport pybayes.wrappers._numpy as np

from pdfs cimport CPdf, Pdf, GaussPdf, EmpPdf, MarginalizedEmpPdf


cdef class Filter(object):
    cpdef bint bayes(self, np.ndarray yt, np.ndarray ut = *) except False
    cpdef Pdf posterior(self)
    cpdef double evidence_log(self, np.ndarray yt) except? -1


cdef class KalmanFilter(Filter):
    cdef readonly np.ndarray A, B, C, D, Q, R
    cdef readonly int n, k, j
    cdef readonly GaussPdf P, S

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __copy__(self)

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __deepcopy__(self, memo)

    @cython.locals(K = np.ndarray)
    cpdef bint bayes(self, np.ndarray yt, np.ndarray ut = *) except False


cdef class ParticleFilter(Filter):
    cdef readonly CPdf p_xt_xtp, p_yt_xt
    cdef readonly EmpPdf emp


cdef class MarginalizedParticleFilter(Filter):
    cdef readonly CPdf p_bt_btp
    cdef readonly np.ndarray kalmans  # dtype=KalmanFilter
    cdef readonly MarginalizedEmpPdf memp

    @cython.locals(kalman = KalmanFilter)
    cpdef bint bayes(self, np.ndarray yt, np.ndarray ut = *) except False

    cpdef bint _resample(self) except False
