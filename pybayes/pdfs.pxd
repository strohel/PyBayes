# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

import cython

cimport numpy as np  # any, array, asarray, diag, dot
#cimport numpy.linalg  # cholesky # numpy.linalg does not (yet) have pxd
#cimport numpy.random  # normal # numpy.random does not (yet) have pxd

cdef class Pdf:

    cpdef np.ndarray[dtype=np.float64_t, ndim=1] mean(self)

    cpdef np.ndarray[dtype=np.float64_t, ndim=1] variance(self)

    #cpdef eval_log(self, np.ndarray[dtype=np.float64, ndim=1] x)  # TODO: return type

    cpdef np.ndarray[dtype=np.float64_t, ndim=1] sample(self)


cdef class GaussPdf(Pdf):

    cdef readonly np.ndarray mu  # TODO: np.ndarray[dtype=np.float64, ndim=1] once permitted by cython
    cdef readonly np.ndarray R  # TODO: np.ndarray[dtype=np.float64, ndim=2] once permitted by cython

    #def __init__(self, np.ndarray[dtype=np.float64, ndim=1] mean,  # init cannot be C-ed
    #             np.ndarray[dtype=np.float64, ndim=2] covariance)

    cpdef np.ndarray[dtype=np.float64_t, ndim=1] mean(self)

    cpdef np.ndarray[dtype=np.float64_t, ndim=1] variance(self)

    #cpdef eval_log(self, x):  # TODO!

    #@cython.locals(z = np.ndarray[dtype=np.float64, ndim=1])  # TODO syntax?
    @cython.locals(z = np.ndarray)
    cpdef np.ndarray[dtype=np.float64, ndim=1] sample(self)
