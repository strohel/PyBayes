# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Definitions for wrapper around numpy - cython version"""

from numpy cimport ndarray


cdef ndarray dot(ndarray a, ndarray b)

cdef double dotvv(ndarray a, ndarray b) except? 0

cdef ndarray inv(ndarray A)
