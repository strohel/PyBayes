# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for kalman filters"""

import unittest as ut
import os.path
import time

import numpy as np
from scipy.io import loadmat, savemat

import pybayes as pb

class TestKalman(ut.TestCase):

    def setUp(self):
        pass

    def test_bayes(self):
        file = os.path.join(os.path.dirname(__file__), "test_kalman_data.mat")
        file_res = os.path.join(os.path.dirname(__file__), "test_kalman_res.mat")

        d = loadmat(file, struct_as_record=True, mat_dtype=True)
        mu0 = np.reshape(d.pop('mu0'), (-1,))  # otherwise we would get 2D array of shape (1xN)
        P0 = d.pop('P0')
        y = d.pop('y')
        u = d.pop('u')

        gauss = pb.pdfs.GaussPdf(mu0, P0)
        kalman = pb.kalman.Kalman(d['A'], d['B'], d['C'], d['D'], d['Q'], d['R'], gauss)

        N = y.shape[1]
        Mu_py = np.zeros((mu0.shape[0], N))

        start = np.array([time.time(), time.clock()])
        for t in xrange(1, N):  # the 1 start offset is intentional
            Mu_py[:,t] = kalman.bayes(y[:,t], u[:,t])
        spent = np.array([time.time(), time.clock()]) - start

        savemat(file_res, {"Mu_py":Mu_py, "exec_time_pybayes":spent[0]}, oned_as='row')
