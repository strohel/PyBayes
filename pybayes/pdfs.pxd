# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

cimport cython
from numpywrap cimport *


cdef class RVComp(object):
    cdef readonly int dimension
    cdef public str name


cdef class RV(object):
    cdef readonly int dimension
    cdef public str name
    cdef readonly list components

    cpdef bint contains(self, RVComp component) except? False
    cpdef bint contains_all(self, components) except? False

    cpdef bint _add_component(self, RVComp component) except False  # cython bug: can be cdef


cdef class CPdf(object):
    cdef public RV rv, cond_rv

    cpdef int shape(self) except -1
    cpdef int cond_shape(self) except -1
    cpdef ndarray mean(self, ndarray cond = *)
    cpdef ndarray variance(self, ndarray cond = *)
    cpdef double eval_log(self, ndarray x, ndarray cond = *) except? -1
    cpdef ndarray sample(self, ndarray cond = *)
    cpdef ndarray samples(self, int n, ndarray cond = *)

    cpdef bint _check_cond(self, ndarray cond) except False  # is internal to PyBayes, thus can be cdef TODO: cython bug
    cpdef bint _check_x(self, ndarray x) except False  # ditto
    cpdef bint _set_rvs(self, RV rv, RV cond_rv) except False  # ditto


cdef class Pdf(CPdf):
    pass


cdef class UniPdf(Pdf):
    cdef public ndarray a, b  # dtype=double


cdef class GaussPdf(Pdf):
    cdef public ndarray mu, R  # dtype=double

    @cython.locals(log_norm = double, log_val = double)
    cpdef double eval_log(self, ndarray x, ndarray cond = *) except? -1

    @cython.locals(z = ndarray)
    cpdef ndarray sample(self, ndarray cond = *)


cdef class EmpPdf(Pdf):
    cdef public ndarray particles  # dtype=double, ndims=2
    cdef public ndarray weights  # dtype=double, ndims=1


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


cdef class MLinGaussCPdf(CPdf):
    cdef public ndarray A, b  # dtype=double
    cdef readonly GaussPdf gauss


cdef class LinGaussCPdf(CPdf):
    cdef public double a, b, c, d
    cdef GaussPdf gauss


cdef class ProdCPdf(CPdf):
    cdef readonly ndarray factors  # dtype=CPdf
    cdef readonly list in_indeces, out_indeces  # dtype=ndarray of ints
    cdef readonly int _shape, _cond_shape
