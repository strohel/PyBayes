# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

cimport cython
from numpywrap cimport *


cdef class Pdf:

    cpdef tuple shape(self)

    cpdef ndarray mean(self)

    cpdef ndarray variance(self)

    cpdef double eval_log(self, ndarray x) except? -1

    cpdef ndarray sample(self)


cdef class GaussPdf(Pdf):

    cdef public ndarray mu
    cdef public ndarray R

    @cython.locals(log_norm = double, log_val = double)
    cpdef double eval_log(self, ndarray x) except? -1

    @cython.locals(z = ndarray)
    cpdef ndarray sample(self)
