# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

cimport cython
from numpywrap cimport *


cdef class CPdf:
    cpdef int shape(self) except -1
    cpdef int cond_shape(self) except -1
    cpdef ndarray cmean(self, ndarray cond)
    cpdef ndarray cvariance(self, ndarray cond)
    cpdef double ceval_log(self, ndarray x, ndarray cond) except? -1
    cpdef ndarray csample(self, ndarray cond)


cdef class Pdf(CPdf):
    cpdef ndarray mean(self)
    cpdef ndarray variance(self)
    cpdef double eval_log(self, ndarray x) except? -1
    cpdef ndarray sample(self)


cdef class UniPdf(Pdf):
    cdef public double a, b


cdef class GaussPdf(Pdf):
    cdef public ndarray mu, R

    @cython.locals(log_norm = double, log_val = double)
    cpdef double eval_log(self, ndarray x) except? -1

    @cython.locals(z = ndarray)
    cpdef ndarray sample(self)
