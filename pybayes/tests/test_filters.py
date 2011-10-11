# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for kalman"""

from copy import copy, deepcopy

import numpy as np

import pybayes as pb
from support import PbTestCase, stochastic


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
            setup[arg] = 125.65
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
        for i in range(4):
            k.bayes(y[i], u[i])
            mu = k.posterior().mu
            self.assertApproxEqual(mu, exp_mu[i])

    def test_copy(self):
        """Test that copying KF works as expected"""
        o = pb.KalmanFilter(**self.setup_1)  # original
        c = copy(o)  # copy
        self.assertEqual(type(c), type(o))

        self.assertTrue(id(o) != id(c))
        self.assertTrue(id(o.A) == id(c.A))
        self.assertTrue(id(o.B) == id(c.B))
        self.assertTrue(id(o.C) == id(c.C))
        self.assertTrue(id(o.D) == id(c.D))
        self.assertTrue(id(o.Q) == id(c.Q))
        self.assertTrue(id(o.R) == id(c.R))
        self.assertTrue(id(o.n) == id(c.n))
        self.assertTrue(id(o.k) == id(c.k))
        self.assertTrue(id(o.j) == id(c.j))
        self.assertTrue(id(o.P) == id(c.P))
        self.assertTrue(id(o.S) == id(c.S))

    def test_deepcopy(self):
        """Test that deep copying KF works as expected"""
        o = pb.KalmanFilter(**self.setup_2)  # original
        c = deepcopy(o)  # copy
        self.assertEqual(type(c), type(o))

        self.assertTrue(id(o) != id(c))
        for (a, b) in [(o.A, c.A), (o.B, c.B), (o.C, c.C), (o.D, c.D), (o.Q, c.Q), (o.R, c.R)]:
            self.assertArraysEqualNotSame(a, b)
        # n, k, j do not need to be different as they are immutable
        self.assertEqual(o.n, c.n)
        self.assertEqual(o.k, c.k)
        self.assertEqual(o.j, c.j)
        self.assertTrue(id(o.P) != id(c.P))
        self.assertArraysEqualNotSame(o.P.mu, c.P.mu)  # this is better tested in
        self.assertArraysEqualNotSame(o.P.R, c.P.R)  # GaussPdf deepcopy test, but wont hurt here
        self.assertTrue(id(o.S) != id(c.S))


class testParticleFilter(PbTestCase):
    """Tests for particle filter"""

    def setUp(self):
        init_pdf = pb.UniPdf(np.array([-5.]), np.array([5.]))
        p_xt_xtp = pb.MLinGaussCPdf(np.array([[2.]]), np.array([[1.]]), np.array([0.]))
        p_yt_xt = pb.MLinGaussCPdf(np.array([[1.]]), np.array([[1.]]), np.array([0.]))

        self.pf = pb.ParticleFilter(20, init_pdf, p_xt_xtp, p_yt_xt)

    def test_init(self):
        self.assertEqual(type(self.pf), pb.ParticleFilter)

    def test_bayes(self):
        # TODO: this test currently does little to verify that PF gives appropriate results
        #np.set_printoptions(linewidth=120, precision=2, suppress=True)
        for i in range(20):
            self.pf.bayes(np.array([i], dtype=float))
            pdf = self.pf.posterior()
            #print "observation, mean:", i, pdf.mean()[0]

    @stochastic
    def test_bayes_with_cond(self):
        """Test that ParticleFilter.bayes() with cond specified works."""
        E = np.array([[1.]])
        o = np.array([0.])
        init_pdf = pb.UniPdf(np.array([-5.]), np.array([5.]))
        p_xt_xtp = pb.LinGaussCPdf(1., 1., 1., 0.)
        p_yt_xt = pb.MLinGaussCPdf(E, E, o)

        pf = pb.ParticleFilter(100, init_pdf, p_xt_xtp, p_yt_xt)
        #np.set_printoptions(linewidth=120, precision=2, suppress=True)
        for i in range(20):
            pf.bayes(np.array([i], dtype=float), np.array([1.]))
            pdf = pf.posterior()
            self.assertTrue(abs(pdf.mean()[0] - i) < 0.3)
            #print "observation, mean:", i, pdf.mean()[0]
