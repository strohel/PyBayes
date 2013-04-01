# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

cimport ceygen.dtype as d
# cython workaround: cannot import *
from numpy import any, array, asarray, concatenate, cumsum, diag, exp, eye, ones, prod


cdef double[:] vector(int size):
    return d.vector(size, <double *> 0)

cdef int[:] index_vector(int size):
    return d.vector(size, <int *> 0)

cdef int[:] index_range(int start, int stop):
    cdef int[:] v = d.vector(stop - start, <int *> 0)
    for i in range(stop - start):
        v[i] = start + i
    return v

cdef double[:, :] matrix(int rows, int cols):
    return d.matrix(rows, cols, <double *> 0)

cdef double[:] zeros(int size):
    ret = vector(size)
    ret[:] = 0
    return ret

cdef bint reindex_vv(reindexable[:] data, int[:] indices) except False:
    assert data.shape[0] == indices.shape[0]
    cdef int newi
    datacopy = data.copy()
    for i in range(data.shape[0]):
        newi = indices[i]
        assert newi >= 0 and newi < data.shape[0]
        data[i] = datacopy[newi]
    return True


cdef bint reindex_mv(reindexable[:, :] data, int[:] indices) except False:
    assert data.shape[0] == indices.shape[0]
    cdef int newi
    datacopy = data.copy()
    for i in range(data.shape[0]):
        newi = indices[i]
        assert newi >= 0 and newi < data.shape[0]
        data[i, :] = datacopy[newi, :]
    return True

cdef double[:] take_vv(double[:] data, int[:] indices, double[:] out = None):
    if out is None:
        out = d.vector(indices.shape[0], <double *> 0)
    else:
        assert out.shape[0] >= indices.shape[0]
    for i in range(indices.shape[0]):
        out[i] = data[indices[i]]
    return out

cdef bint put_vv(double[:] out, int[:] indices, double[:] data) except False:
    assert data.shape[0] >= indices.shape[0]
    for i in range(indices.shape[0]):
        out[indices[i]] = data[i]
    return True
