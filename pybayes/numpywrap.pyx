# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - cython version"""

# import and flatten numpy types and functions, override some as needed.
# this file is special - it is used only in cython build, this can contain code
# not callable from python etc.

from numpy import any as np_any, array, asarray, diag, dot as dot_, ndarray
from numpy.linalg import cholesky, inv
from numpy.random import normal

cdef ndarray dot(ndarray a, ndarray b):
    print("special dot!")
    return dot_(a, b)
