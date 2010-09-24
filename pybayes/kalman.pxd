# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

import cython

# TODO: cimport


cdef class Kalman:

    cpdef np.ndarray A  # TODO: np.ndarray[dtype=np.float64, ndim=2] once permitted by cython
    cpdef np.ndarray B  # ditto
    cpdef np.ndarray C  # ditto
    cpdef np.ndarray D  # ditto
    cpdef np.ndarray Q  # ditto
    cpdef np.ndarray R  # ditto
