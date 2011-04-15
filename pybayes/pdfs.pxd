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

    @cython.locals(ret = RV)
    cpdef RV __copy__(self)

    @cython.locals(ret = RV)
    cpdef RV __deepcopy__(self, memo)

    cpdef bint contains(self, RVComp component) except? False
    cpdef bint contains_all(self, test_components) except? False
    cpdef bint contains_any(self, test_components) except? False

    cdef bint _add_component(self, RVComp component) except False


cdef class CPdf(object):
    cdef public RV rv, cond_rv

    cpdef int shape(self) except -1
    cpdef int cond_shape(self) except -1
    cpdef ndarray mean(self, ndarray cond = *)
    cpdef ndarray variance(self, ndarray cond = *)
    cpdef double eval_log(self, ndarray x, ndarray cond = *) except? -1
    cpdef ndarray sample(self, ndarray cond = *)
    cpdef ndarray samples(self, int n, ndarray cond = *)

    cdef bint _check_cond(self, ndarray cond) except False
    cdef bint _check_x(self, ndarray x) except False
    cdef bint _set_rvs(self, RV rv, RV cond_rv) except False


cdef class Pdf(CPdf):
    pass


cdef class UniPdf(Pdf):
    cdef public ndarray a, b  # dtype=double


cdef class AbstractGaussPdf(Pdf):
    cdef public ndarray mu, R  # dtype=double

    @cython.locals(ret = AbstractGaussPdf)
    cpdef AbstractGaussPdf __copy__(self)

    @cython.locals(ret = AbstractGaussPdf)
    cpdef AbstractGaussPdf __deepcopy__(self, memo)


cdef class GaussPdf(AbstractGaussPdf):

    @cython.locals(log_norm = double, log_val = double)
    cpdef double eval_log(self, ndarray x, ndarray cond = *) except? -1

    @cython.locals(z = ndarray)
    cpdef ndarray sample(self, ndarray cond = *)


cdef class LogNormPdf(AbstractGaussPdf):
    pass  # everything inherited from AbstractGaussPdf


cdef class AbstractEmpPdf(Pdf):
    cdef public ndarray weights  # dtype=double, ndims=1

    @cython.locals(wsum = double)
    cpdef bint normalise_weights(self) except False

    @cython.locals(n = int, cum_weights = ndarray, u = ndarray, baby_indeces = ndarray, j = int)
    cpdef ndarray get_resample_indices(self)


cdef class EmpPdf(AbstractEmpPdf):
    cdef public ndarray particles  # dtype=double, ndims=2

    cpdef bint resample(self) except False


cdef class MarginalizedEmpPdf(AbstractEmpPdf):
    cdef public ndarray gausses  # dtype=GaussPdf, ndims=1
    cdef public ndarray particles  # dtype=double, ndims=2
    cdef public int _gauss_shape, _part_shape


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
    cdef AbstractGaussPdf gauss

    cdef bint _set_mean(self, ndarray cond) except False


cdef class LinGaussCPdf(CPdf):
    cdef public double a, b, c, d
    cdef AbstractGaussPdf gauss

    @cython.locals(c0 = double, c1 = double)
    cdef bint _set_gauss_params(self, ndarray cond) except False


cdef class GaussCPdf(CPdf):
    cdef int _shape, _cond_shape
    cdef public object f, g  # callables
    cdef AbstractGaussPdf gauss

    cdef bint _set_gauss_params(self, ndarray cond) except False


cdef class ProdCPdf(CPdf):
    cdef readonly ndarray factors  # dtype=CPdf
    cdef readonly list in_indeces, out_indeces  # dtype=ndarray of ints
    cdef int _shape, _cond_shape
