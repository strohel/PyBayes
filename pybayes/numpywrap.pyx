# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import and flatten numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

from numpy cimport int
from numpy import any as np_any, array, asarray, diag, ndarray, empty
from numpy.linalg import cholesky
from numpy.random import normal

cimport tokyo as t


cdef ndarray dot(ndarray a, ndarray b):
    if a.ndim == 2:
        if b.ndim == 1:
            return t.dgemv(a, b)
        if b.ndim == 2:
            return t.dgemm(a, b)
    raise ValueError("I can only handle matrix*vector and matrix*matrix!")

cdef ndarray inv(ndarray A):
    p = empty((A.shape[0],), dtype=int)  # TODO: faster alt
    R = A.copy()  # TODO: faster cython version

    if t.dgetrf(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetrf)")
    if t.dgetri(R, p) != 0:
        raise ValueError("A is singular or invalid argument passed (dgetri)")
    return R
