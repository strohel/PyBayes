# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for pdfs.py"""

cimport cython
from numpy cimport ndarray


cdef class Pdf:

    cpdef tuple shape(self)

    cpdef ndarray mean(self)

    cpdef ndarray variance(self)

    cpdef object eval_log(self, ndarray x)  # TODO: dtype of all arrays

    cpdef ndarray sample(self)


cdef class GaussPdf(Pdf):

    cdef public ndarray mu  # TODO: readonly
    cdef public ndarray R  # TODO: readonly

    cpdef tuple shape(self)

    cpdef ndarray mean(self)

    cpdef ndarray variance(self)

    #cpdef eval_log(self, x):  # TODO

    @cython.locals(z = ndarray)
    cpdef ndarray sample(self)
