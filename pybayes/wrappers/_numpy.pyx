# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

from numpy cimport *
# cython workaround: cannot import *
from numpy import any, arange, array, asarray, cumsum, diag, empty, exp, eye, ones, prod, sum, zeros

cimport tokyo as t


import_array()  # needed to call any PyArray functions

cdef ndarray dot(ndarray a, ndarray b):
    cdef t.CBLAS_TRANSPOSE trans_a, trans_b  # stored directly, or transposed?
    cdef ndarray ret
    cdef npy_intp shape[2]
    cdef int lda, ldb

    if a is None:
        raise TypeError("a must be numpy.ndarray")
    if b is None:
        raise TypeError("b must be numpy.ndarray")
    if a.descr.type_num != NPY_DOUBLE:
        raise ValueError("a is not of type double")
    if b.descr.type_num != NPY_DOUBLE:
        raise ValueError("b is not of type double")

    if a.ndim != 2:
        raise ValueError("I can only handle matrix*vector and matrix*matrix!")

    if a.shape[1] != b.shape[0]:
        raise ValueError("a columns != b rows")

    if PyArray_ISCARRAY_RO(a):
        trans_a = t.CblasNoTrans
        ldb = a.shape[0]  # abused for number of A rows
        lda = a.shape[1]
    elif PyArray_ISFARRAY_RO(a):
        trans_a = t.CblasTrans
        ldb = a.shape[1]  # abused for number of A.T rows
        lda = a.shape[0]
    else:
        raise ValueError("a must be C contiguos or F contiguos (ro)")

    if b.ndim == 1:  # matrix * vector
        if not PyArray_ISCARRAY_RO(b):
            raise ValueError("b must be C Contiguos (ro) array")

        shape[0] = a.shape[0];  # prepare shape to pass to dgemv_
        ret = PyArray_EMPTY(1, shape, NPY_DOUBLE, 0)  # create empty array for result of right dimension

        if a.shape[0] > 0 and a.shape[1] > 0:  # otherwise BLAS may fail
            t.dgemv_(t.CblasRowMajor, trans_a, ldb, lda, 1.0, <double*> a.data,
                     lda, <double*> b.data, 1, 0.0, <double*> ret.data, 1)
        return ret

    if b.ndim == 2:  # matrix * matrix
        if PyArray_ISCARRAY_RO(b):
            trans_b = t.CblasNoTrans
            ldb = b.shape[1]
        elif PyArray_ISFARRAY_RO(b):
            trans_b = t.CblasTrans
            ldb = b.shape[0]
        else:
            raise ValueError("b must be C contiguos or F contiguos (ro)")

        shape[0] = a.shape[0]  # prepare shape to pass to dgemm_
        shape[1] = b.shape[1]
        ret = PyArray_EMPTY(2, shape, NPY_DOUBLE, 0)  # allocate retsult matrix

        t.dgemm_(t.CblasRowMajor, trans_a, trans_b, a.shape[0], b.shape[1], a.shape[1],
                 1.0, <double*> a.data, lda, <double*> b.data, ldb,
                 0.0, <double*> ret.data, b.shape[1])
        return ret

# this is defined separately because of different return type
cdef double dotvv(ndarray a, ndarray b) except? -1:
    if a is None:
        raise TypeError("a must be numpy.ndarray")
    if b is None:
        raise TypeError("b must be numpy.ndarray")
    if a.descr.type_num != NPY_DOUBLE:
        raise ValueError("a is not of type double")
    if b.descr.type_num != NPY_DOUBLE:
        raise ValueError("b is not of type double")
    if a.ndim != 1:
        raise ValueError("a is not a vector")
    if b.ndim != 1:
        raise ValueError("b is not a vector")
    if a.shape[0] != b.shape[0]:
        raise ValueError("a columns != b columns")
    if not PyArray_ISCARRAY_RO(a):
        raise ValueError("a is not C contiguos (ro)")
    if not PyArray_ISCARRAY_RO(b):
        raise ValueError("b is not C contiguos (ro)")

    return t.ddot_(a.shape[0], <double*> a.data, 1, <double*> b.data, 1)
