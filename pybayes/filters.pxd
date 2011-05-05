# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

cimport cython

cimport pybayes.wrappers._linalg as linalg
cimport pybayes.wrappers._numpy as np

from pdfs cimport CPdf, Pdf, GaussPdf, EmpPdf, MarginalizedEmpPdf


cdef class Filter(object):
    cpdef bint bayes(self, np.ndarray yt, np.ndarray cond = *) except False
    cpdef Pdf posterior(self)
    cpdef double evidence_log(self, np.ndarray yt) except? -1


cdef class KalmanFilter(Filter):
    cdef readonly np.ndarray A, C, Q, R
    cdef readonly int n, j
    cdef readonly GaussPdf P, S

    cpdef bint _check_matrix(self, name, matrix) except False

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __copy__(self)

    @cython.locals(ret = KalmanFilter)
    cpdef KalmanFilter __deepcopy__(self, memo)

    cpdef bint _cond_preprocess(self, np.ndarray cond) except False
    cpdef bint _cond_predict(self, np.ndarray cond) except False
    cpdef bint _cond_update(self, np.ndarray cond) except False

    @cython.locals(K = np.ndarray)
    cpdef bint bayes(self, np.ndarray yt, np.ndarray cond = *) except False


cdef class ControlKalmanFilter(KalmanFilter):
    cdef readonly np.ndarray B, D
    cdef readonly k

    @cython.locals(ret = ControlKalmanFilter)
    cpdef KalmanFilter __copy__(self)

    @cython.locals(ret = ControlKalmanFilter)
    cpdef KalmanFilter __deepcopy__(self, memo)


cdef class QRKalmanFilter(KalmanFilter):
    pass


cdef class ParticleFilter(Filter):
    cdef readonly CPdf p_xt_xtp, p_yt_xt
    cdef readonly EmpPdf emp


cdef class MarginalizedParticleFilter(Filter):
    cdef readonly CPdf p_bt_btp
    cdef readonly np.ndarray kalmans  # dtype=QRKalmanFilter
    cdef readonly MarginalizedEmpPdf memp

    @cython.locals(kalman = KalmanFilter)
    cpdef bint bayes(self, np.ndarray yt, np.ndarray cond = *) except False

    cpdef bint _resample(self) except False
