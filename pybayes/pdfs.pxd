# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

import cython
cimport numpy as np  # any, array, asarray, diag, dot
#cimport numpy.linalg  # cholesky # numpy.linalg does not (yet) have pxd
#cimport numpy.random  # normal # numpy.random does not (yet) have pxd

from utils cimport data_t


cdef class Pdf:

    cpdef np.ndarray mean(self)

    cpdef np.ndarray variance(self)

    cpdef data_t eval_log(self, np.ndarray x) except *

    cpdef np.ndarray sample(self)


cdef class GaussPdf(Pdf):

    cdef public np.ndarray mu  # TODO: readonly
    cdef public np.ndarray R  # TODO: readonly

    #def __init__(self, np.ndarray mean, np.ndarray covariance)  # init cannot be cpdef-ed

    cpdef np.ndarray mean(self)

    cpdef np.ndarray variance(self)

    #cpdef eval_log(self, x):  # TODO

    @cython.locals(z = np.ndarray)
    cpdef np.ndarray sample(self)
