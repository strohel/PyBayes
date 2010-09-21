# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for kalman filters"""

import unittest as ut
import os.path
import time

import numpy as np
from scipy.io import loadmat

import pybayes as pb

class TestKalman(ut.TestCase):

    def setUp(self):
        pass

    def test_bayes(self):
        file = os.path.join(os.path.dirname(__file__), "test_kalman_data.mat")
        d = loadmat(file, struct_as_record=True, mat_dtype=True)
        mu0 = np.reshape(d['mu0'], -1)  # otherwise we would get 2D array of shape (1xN)

        gauss = pb.pdfs.GaussPdf(mu0, d['P0'])
        kalman = pb.kalman.Kalman(d['A'], np.squeeze(d['B'].T), d['C'], d['D'], d['Q'], d['R'], gauss)

        y = np.squeeze(d['y'])  # to prevent 2D array (1xN)
        u = np.squeeze(d['u'])  # ditto
        N = y.shape[0]
        Mu_py = np.zeros((mu0.shape[0], N))

        #print "y, u, N, mu0, Mu_py:", y, u, N, mu0, Mu_py

        start = np.array([time.time(), time.clock()])
        for t in xrange(1, N):  # the 1 start offset is intentional
            temp = kalman.bayes(y[t], u[t])
            Mu_py[:,t] = 0
        spent = np.array([time.time(), time.clock()]) - start

        print "time spent [real, cpu]:", spent
