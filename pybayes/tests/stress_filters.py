# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Stresses for kalman filters"""

import os.path
import time

import numpy as np
from scipy.io import loadmat, savemat

import pybayes.filters as filters
import pybayes.pdfs as pdfs

def run_kalman_on_mat_data(input_file, output_file):
    d = loadmat(input_file, struct_as_record=True, mat_dtype=True)

    mu0 = np.reshape(d.pop('mu0'), (-1,))  # otherwise we would get 2D array of shape (1xN)
    P0 = d.pop('P0')
    y = d.pop('y').T
    u = d.pop('u').T

    gauss = pdfs.GaussPdf(mu0, P0)
    kalman = filters.KalmanFilter(d['A'], d['B'], d['C'], d['D'], d['Q'], d['R'], gauss)

    N = y.shape[0]
    n = mu0.shape[0]
    Mu_py = np.zeros((N, n))

    start = np.array([time.time(), time.clock()])
    for t in xrange(1, N):  # the 1 start offset is intentional
        Mu_py[t] = kalman.bayes(y[t], u[t]).mu
    spent = np.array([time.time(), time.clock()]) - start

    Mu_py = Mu_py.T
    print("time spent: " + str(spent[0]) + "s real time; " + str(spent[1]) + "s CPU time")
    savemat(output_file, {"Mu_py":Mu_py, "exec_time_pybayes":spent[0]}, oned_as='row')

def main(options):
    input_file = options.datadir + "/stress_kalman_data.mat"
    output_file = "stress_kalman_res.mat"

    run_kalman_on_mat_data(input_file, output_file)
