# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

cimport cython
from numpywrap cimport *


cdef class CPdf(object):
    cpdef int shape(self) except -1
    cpdef int cond_shape(self) except -1
    cpdef ndarray mean(self, ndarray cond = *)
    cpdef ndarray variance(self, ndarray cond = *)
    cpdef double eval_log(self, ndarray x, ndarray cond = *) except? -1
    cpdef ndarray sample(self, ndarray cond = *)

    cpdef bint check_cond(self, ndarray cond) except False  # is internal to PyBayes, thus can be cdef TODO: cython bug


cdef class Pdf(CPdf):
    pass


cdef class UniPdf(Pdf):
    cdef public ndarray a, b  # dtype=double


cdef class ProdPdf(Pdf):
    cdef readonly ndarray factors  # dtype=Pdf
    cdef readonly ndarray shapes  # dtype=int
    cdef int _shape

    @cython.locals(curr = int, i = int, ret = ndarray)
    cpdef ndarray mean(self, ndarray cond = *)

    @cython.locals(curr = int, i = int, ret = ndarray)
    cpdef ndarray variance(self, ndarray cond = *)

    @cython.locals(curr = int, i = int, ret = double)
    cpdef double eval_log(self, ndarray x, ndarray cond = *)

    @cython.locals(curr = int, i = int, ret = ndarray)
    cpdef ndarray sample(self, ndarray cond = *)


cdef class GaussPdf(Pdf):
    cdef public ndarray mu, R  # dtype=double

    @cython.locals(log_norm = double, log_val = double)
    cpdef double eval_log(self, ndarray x, ndarray cond = *) except? -1

    @cython.locals(z = ndarray)
    cpdef ndarray sample(self, ndarray cond = *)


cdef class MLinGaussPdf(CPdf):
    cdef public ndarray A, b  # dtype=double
    cdef readonly GaussPdf gauss
