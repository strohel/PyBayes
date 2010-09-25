# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Cython augmentation file for kalman.py"""

import cython

cimport numpy as np

cimport pdfs


cdef class Kalman:

    cdef public np.ndarray A, B, C, D, Q, R
    cdef readonly int n, k, j
    cdef readonly pdfs.GaussPdf P, S
    cdef bint _bayes_type_check


    #def __init__(self, A, B, C, D, Q, R, state_pdf)
        #n, k, j
        #self.P = state_pdf
        #self.S = GaussPdf()  # observation probability density function
        #self._bayes_type_check = True  # whether to check arguments in bayes() method

    @cython.locals(K = np.ndarray)
    cpdef np.ndarray bayes(self, np.ndarray yt, np.ndarray ut)
        #K
