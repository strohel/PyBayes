# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

cimport ceygen.core as c
from numpy cimport *
# cython workaround: cannot import *
from numpy import any, arange, array, asarray, concatenate, cumsum, diag, empty, exp, eye, ones
from numpy import prod, sum, zeros


import_array()  # needed to call any PyArray functions

cdef ndarray dot(ndarray a, ndarray b):
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

    if a.ndim == 1:
        if b.ndim == 1:
            raise ValueError("Use dotvv for vector * vector dot product")
        elif b.ndim == 2:
            shape[0] = b.shape[1]  # prepare shape to pass to PyArray_EMPTY
            ret = PyArray_EMPTY(1, shape, NPY_DOUBLE, 0)  # shortcut to np.empty()
            c.dot_vm(a, b, ret)
            return ret
        else:
            raise ValueError("I cannot handle ndarrays with ndim > 2")
    elif a.ndim == 2:
        if b.ndim == 1:
            shape[0] = a.shape[0]
            ret = PyArray_EMPTY(1, shape, NPY_DOUBLE, 0)
            c.dot_mv(a, b, ret)
            return ret
        elif b.ndim == 2:
            shape[0] = a.shape[0]
            shape[1] = b.shape[1]
            ret = PyArray_EMPTY(2, shape, NPY_DOUBLE, 0)
            c.dot_mm(a, b, ret)
            return ret
        else:
            raise ValueError("I cannot handle ndarrays with ndim > 2")
    else:
        raise ValueError("I cannot handle ndarrays with ndim > 2")


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
    return c.dot_vv(a, b)
