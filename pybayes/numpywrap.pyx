# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import and flatten numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

from numpy cimport import_array, int, npy_intp, NPY_DOUBLE, PyArray_EMPTY, PyArray_ISCARRAY_RO, PyArray_ISFARRAY_RO
from numpy import any as np_any, array, asarray, diag, empty, zeros
from numpy.linalg import cholesky, slogdet
from numpy.random import normal, uniform

cimport tokyo as t


import_array()  # needed to call any PyArray functions

cdef ndarray dot(ndarray a, ndarray b):
    cdef t.CBLAS_ORDER order_a  # needed for matrix * vector
    cdef t.CBLAS_TRANSPOSE trans_a, trans_b  # needed for matrix * matrix
    cdef ndarray ret
    cdef npy_intp shape[2]

    if a is None:
        raise TypeError("a must be numpy.ndarray")
    if b is None:
        raise TypeError("b must be numpy.ndarray")
    if a.descr.type_num != NPY_DOUBLE:
        raise ValueError("a is not of type double")
    if b.descr.type_num != NPY_DOUBLE:
        raise ValueError("b is not of type double")

    if PyArray_ISCARRAY_RO(a):
        order_a = t.CblasRowMajor
        trans_a = t.CblasNoTrans
    elif PyArray_ISFARRAY_RO(a):
        order_a = t.CblasColMajor
        trans_a = t.CblasTrans
    else:
        raise ValueError("a must be C contiguos or F contiguos (ro)")

    if a.ndim == 2:
        if a.shape[1] != b.shape[0]:
            raise ValueError("a columns != b rows")
        if b.ndim == 1:  # matrix * vector
            if not PyArray_ISCARRAY_RO(b):
                raise ValueError("b must be C Contiguos (ro) array")

            shape[0] = a.shape[0];  # prepare shape to pass to dgemv_
            ret = PyArray_EMPTY(1, shape, NPY_DOUBLE, 0)  # create empty array for result of right dimension

            t.dgemv_(order_a, t.CblasNoTrans, a.shape[0], a.shape[1], 1.0, <double*> a.data,
                     a.shape[1], <double*> b.data, 1, 0.0, <double*> ret.data, 1)
            return ret

        if b.ndim == 2:  # matrix * matrix
            if PyArray_ISCARRAY_RO(b):
                trans_b = t.CblasNoTrans
            elif PyArray_ISFARRAY_RO(b):
                trans_b = t.CblasTrans
            else:
                raise ValueError("b must be C contiguos or F contiguos (ro)")

            shape[0] = a.shape[0]  # prepare shape to pass to dgemm_
            shape[1] = b.shape[1]
            ret = PyArray_EMPTY(2, shape, NPY_DOUBLE, 0)  # allocate retsult matrix

            t.dgemm_(t.CblasRowMajor, trans_a, trans_b, ret.shape[0], ret.shape[1], b.shape[0],
                     1.0, <double*> a.data, a.shape[1], <double*> b.data, b.shape[1],
                     0.0, <double*> ret.data, ret.shape[1])
            return ret
    raise ValueError("I can only handle matrix*vector and matrix*matrix!")

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

cdef ndarray inv(ndarray A):
    p = empty((A.shape[0],), dtype=int)
    R = A.copy()

    if t.dgetrf(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetrf)")
    if t.dgetri(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetri)")
    return R
