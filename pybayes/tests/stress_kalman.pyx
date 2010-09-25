# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Stresses for kalman filters"""

import os.path
import time

cimport cython
import numpy as np
cimport numpy as np
from scipy.io import loadmat, savemat

import pybayes.kalman as kf
cimport pybayes.kalman as kf
import pybayes.pdfs as pdfs
cimport pybayes.pdfs as pdfs

@cython.boundscheck(False)
@cython.wraparound(False)
cdef run_kalman_on_mat_data(input_file, output_file):
    cdef np.ndarray[np.float64_t, ndim=2] y, u, Mu_py
    cdef kf.Kalman kalman
    cdef int t

    d = loadmat(input_file, struct_as_record=True, mat_dtype=True)

    mu0 = np.reshape(d.pop('mu0'), (-1,))  # otherwise we would get 2D array of shape (1xN)
    P0 = d.pop('P0')
    y = d.pop('y').T
    u = d.pop('u').T

    gauss = pdfs.GaussPdf(mu0, P0)
    kalman = kf.Kalman(d['A'], d['B'], d['C'], d['D'], d['Q'], d['R'], gauss)

    N = y.shape[0]
    n = mu0.shape[0]
    Mu_py = np.zeros((N, n))

    start = np.array([time.time(), time.clock()])
    for t in xrange(1, N):  # the 1 start offset is intentional
        Mu_py[t] = kalman.bayes(y[t], u[t])
    spent = np.array([time.time(), time.clock()]) - start

    Mu_py = Mu_py.T
    print("time spent: " + str(spent[0]) + "s real time; " + str(spent[1]) + "s CPU time")
    savemat(output_file, {"Mu_py":Mu_py, "exec_time_pybayes":spent[0]}, oned_as='row')

cpdef main():
    input_file = os.path.join(os.path.dirname(__file__), "stress_kalman_data.mat")
    output_file = os.path.join(os.path.dirname(__file__), "stress_kalman_res.mat")

    run_kalman_on_mat_data(input_file, output_file)
