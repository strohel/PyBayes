# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

cimport cython

cimport pybayes.wrappers._linalg as linalg
cimport pybayes.wrappers._numpy as np


# workarounds for Cython:
ctypedef double[:] double_1D
ctypedef double[:, :] double_2D
ctypedef int[:] int_1D

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
    cpdef double[:] mean(self, double[:] cond = *)
    cpdef double[:] variance(self, double[:] cond = *)
    cpdef double eval_log(self, double[:] x, double[:] cond = *) except? -1
    cpdef double[:] sample(self, double[:] cond = *)
    @cython.locals(ret = double_2D)
    cpdef double[:, :] samples(self, int n, double[:] cond = *)

    cpdef bint _check_cond(self, double[:] cond) except False
    cpdef bint _check_x(self, double[:] x) except False
    cpdef bint _set_rvs(self, int exp_shape, rv, int exp_cond_shape, cond_rv) except False


cdef class Pdf(CPdf):
    cpdef bint _set_rv(self, int exp_shape, rv) except False


cdef class UniPdf(Pdf):
    # intentionally ndarrays
    cdef public np.ndarray a, b


cdef class AbstractGaussPdf(Pdf):
    cdef public double[:] mu
    cdef public double[:, :] R

    @cython.locals(ret = AbstractGaussPdf)
    cpdef AbstractGaussPdf __copy__(self)

    @cython.locals(ret = AbstractGaussPdf)
    cpdef AbstractGaussPdf __deepcopy__(self, memo)


cdef class GaussPdf(AbstractGaussPdf):

    @cython.locals(log_norm = double, distance = double_1D, log_val = double)
    cpdef double eval_log(self, double[:] x, double[:] cond = *) except? -1

    @cython.locals(z = double_1D)
    cpdef double[:] sample(self, double[:] cond = *)


cdef class LogNormPdf(AbstractGaussPdf):
    pass  # everything inherited from AbstractGaussPdf


cdef class TruncatedNormPdf(Pdf):
    cdef public double mu, sigma_sq, a, b

    cdef double _pdf(self, double x) except -1.
    cdef double _cdf(self, double x) except -1.

cdef class GammaPdf(Pdf):
    cdef public double k, theta


cdef class InverseGammaPdf(Pdf):
    cdef public double alpha, beta


cdef class AbstractEmpPdf(Pdf):
    cdef public double[:] weights
    cdef public double[:, :] particles

    @cython.locals(wsum = double)
    cpdef bint normalise_weights(self) except False

    @cython.locals(n = int, cum_weights = double_1D, u = double_1D, baby_indices = int_1D, j = int)
    cpdef int[:] get_resample_indices(self)


cdef class EmpPdf(AbstractEmpPdf):
    cpdef bint resample(self) except False
    cpdef bint transition_using(self, int i, CPdf transition_cpdf) except False


cdef class MarginalizedEmpPdf(AbstractEmpPdf):
    cdef public GaussPdf[:] gausses
    cdef public int _gauss_shape, _part_shape

    @cython.locals(gauss = GaussPdf)
    cpdef double[:] mean(self, double[:] cond = *)

    @cython.locals(nom2 = double_1D, temp = double_1D, gauss = GaussPdf, mean = double_1D, var = double_1D)
    cpdef double[:] variance(self, double[:] cond = *)


cdef class ProdPdf(Pdf):
    cdef readonly Pdf[:] factors
    cdef readonly int[:] shapes
    cdef int _shape

    @cython.locals(curr = int, factor = Pdf, i = int, ret = double_1D)
    cpdef double[:] mean(self, double[:] cond = *)

    @cython.locals(curr = int, factor = Pdf, i = int, ret = double_1D)
    cpdef double[:] variance(self, double[:] cond = *)

    @cython.locals(curr = int, factor = Pdf, i = int, ret = double)
    cpdef double eval_log(self, double[:] x, double[:] cond = *)

    @cython.locals(curr = int, factor = Pdf, i = int, ret = double_1D)
    cpdef double[:] sample(self, double[:] cond = *)

    cpdef int _calculate_shape(self) except -1


cdef class MLinGaussCPdf(CPdf):
    cdef public double[:, :] A
    cdef public double[:] b
    cdef AbstractGaussPdf gauss

    cdef bint _set_mean(self, double[:] cond) except False


cdef class LinGaussCPdf(CPdf):
    cdef public double a, b, c, d
    cdef AbstractGaussPdf gauss

    @cython.locals(c0 = double, c1 = double)
    cdef bint _set_gauss_params(self, double[:] cond) except False


cdef class GaussCPdf(CPdf):
    cdef int _shape, _cond_shape
    cdef public object f, g  # callables
    cdef AbstractGaussPdf gauss

    cdef bint _set_gauss_params(self, double[:] cond) except False


cdef class GammaCPdf(CPdf):
    cdef public double gamma
    cdef public GammaPdf gamma_pdf


cdef class InverseGammaCPdf(CPdf):
    cdef public double gamma
    cdef public InverseGammaPdf igamma_pdf


cdef class ProdCPdf(CPdf):
    cdef readonly CPdf[:] factors
    cdef readonly list in_indeces, out_indeces  # dtype=array of ints
    cdef int _shape, _cond_shape
