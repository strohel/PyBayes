# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# wrappers.linalg test needs this file, because in cython wrappers.linalg, there are only
# cdefs, not cpdefs

cimport cython

cimport pybayes.wrappers._linalg as linalg


ctypedef double[:, :] double_2D

@cython.locals(A = double_2D, iA = double_2D)
cpdef test_inv_func(self)

@cython.locals(arr = double_2D)
cpdef test_slogdet_func(self)

@cython.locals(arr = double_2D, res = double_2D)
cpdef test_cholesky_func(self)
