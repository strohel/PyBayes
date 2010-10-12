# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import and flatten numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

from numpy cimport import_array, int, PyArray_ISCARRAY_RO, PyArray_ISFARRAY_RO
from numpy import any as np_any, array, asarray, diag, ndarray, empty
from numpy.linalg import cholesky
from numpy.random import normal

cimport tokyo as t


import_array()  # needed to call any PyArray functions

cdef ndarray dot(ndarray a, ndarray b):
    cdef t.CBLAS_TRANSPOSE trans_a, trans_b
    cdef ndarray ret
    if a is None:
        raise TypeError("a must be numpy.ndarray")
    if b is None:
        raise TypeError("b must be numpy.ndarray")

    if PyArray_ISCARRAY_RO(a):
        trans_a = t.CblasNoTrans
    elif PyArray_ISFARRAY_RO(a):
        trans_a = t.CblasTrans
    else:
        raise ValueError("a must be C contiguos or F contiguos (ro)")

    if a.ndim == 2:
        if b.ndim == 1:
            if not PyArray_ISCARRAY_RO(b):
                raise ValueError("b must be C Contiguos (ro) array")

            ret = t.dvnewempty(a.shape[0])
            t.dgemv6(trans_a, 1.0, a, b, 0.0, ret)
            return ret

        if b.ndim == 2:
            if PyArray_ISCARRAY_RO(b):
                trans_b = t.CblasNoTrans
            elif PyArray_ISFARRAY_RO(b):
                trans_b = t.CblasTrans
            else:
                raise ValueError("b must be C contiguos or F contiguos (ro)")

            ret = t.dmnewempty(a.shape[0], b.shape[1])
            t.dgemm7(trans_a, trans_b, 1.0, a, b, 0.0, ret)
            return ret
    raise ValueError("I can only handle matrix*vector and matrix*matrix!")

cdef ndarray inv(ndarray A):
    p = empty((A.shape[0],), dtype=int)
    R = A.copy()

    if t.dgetrf(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetrf)")
    if t.dgetri(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetri)")
    return R
