# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

cimport cython
from numpywrap cimport *

from pdfs cimport GaussPdf


cdef class Kalman:

    cdef public ndarray A, B, C, D, Q, R
    cdef readonly int n, k, j
    cdef readonly GaussPdf P, S
    cdef bint _bayes_type_check


    #def __init__(self, A, B, C, D, Q, R, state_pdf)
        #n, k, j
        #self.P = state_pdf
        #self.S = GaussPdf()  # observation probability density function
        #self._bayes_type_check = True  # whether to check arguments in bayes() method

    @cython.locals(K = ndarray)
    cpdef ndarray bayes(self, ndarray yt, ndarray ut)
