# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for kalman"""

import numpy as np

import pybayes as pb
from support import PbTestCase


class TestKalmanFilter(PbTestCase):
    """Tests for kalman filter"""

    def setUp(self):
        # synthetic parameters. May be completely mathematically invalid
        self.setup_1 = {  # n = 2, k = 3, j = 4
            "A":np.array([[1, 2], [3, 4]]),  # n*n
            "B":np.array([[1, 2, 3], [4, 5, 6]]),  # n*k
            "C":np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),  # j*n
            "D":np.array([[1, 2, 3], [5, 6, 7], [9, 1, 2], [2, 3, 4]]),  # j*k
            "Q":np.array([[2, 3], [4, 5]]),  # n*n
            "R":np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [2, 3, 4, 5]]),  # j*j
            "state_pdf":pb.GaussPdf(np.array([1, 2]), np.array([[1, 0], [0, 2]]))  # n
        }
        self.setup_2 = {  # n = 2, k = 1, j = 1
            "A":np.array([[1.0, -0.5],[1.0, 0.0]]),
            "B":np.array([[1.0],[0.1]]),
            "C":np.array([[1.0, 0.0]]),
            "D":np.array([[0.1]]),
            "Q":np.array([[0.2, 0.0],[0.0, 0.2]]),
            "R":np.array([[0.01]]),
            "state_pdf":pb.GaussPdf(np.array([0.0, 0.0]), np.array([[200.0, 0.0],[0.0, 200.0]]))
        }

    def test_init(self):
        k = pb.KalmanFilter(**self.setup_1)
        self.assertEqual(type(k), pb.KalmanFilter)
        l = pb.KalmanFilter(**self.setup_2)
        self.assertEqual(type(l), pb.KalmanFilter)

    def test_invalid_init(self):
        args = ["A", "B", "C", "D", "Q", "R", "state_pdf"]

        # invalid type:
        for arg in args:
            setup = self.setup_1.copy()
            setup[arg] = np.mat([[1,2],[3,4]])
            self.assertRaises(TypeError, pb.KalmanFilter, **setup)

        # invalid dimension
        del args[6]  # remove state_pdf
        for arg in args:
            setup = self.setup_1.copy()
            setup[arg] = np.array([[1],[2]])
            self.assertRaises(ValueError, pb.KalmanFilter, **setup)
        gauss = pb.GaussPdf(np.array([1]), np.array([[1]]))
        setup = self.setup_1.copy()
        setup['state_pdf'] = gauss
        self.assertRaises(ValueError, pb.KalmanFilter, **setup)

    def test_bayes(self):
        k = pb.KalmanFilter(**self.setup_2)
        y = np.array([[4.1], [-0.2], [1.4], [-2.1]])
        u = np.array([[4.8], [-0.3], [1.1], [-1.8]])
        exp_mu = np.array([
            [ 3.62004716, -0.46320771],
            [-0.16638519,  3.58787721],
            [ 1.21108425,  0.0224309 ],
            [-1.87141692,  0.98517451]
        ])
        for i in xrange(4):
            mu = k.bayes(y[i], u[i]).mu
            self.assertApproxEqual(mu, exp_mu[i])


class testParticleFilter(PbTestCase):
    """Tests for particle filter"""

    def test_init(self):
        init_pdf = pb.UniPdf(np.array([-5.]), np.array([5.]))
        p_xt_xtp = pb.MLinGaussCPdf(np.array([[1.]]), np.array([[1.]]), np.array([0.]))
        p_xt_yt = pb.MLinGaussCPdf(np.array([[10.]]), np.array([[1.]]), np.array([0.]))

        self.pf = pb.ParticleFilter(5, init_pdf, p_xt_xtp, p_xt_yt)
