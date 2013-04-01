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
            "A":np.array([[1., 2], [3, 4]]),  # n*n
            "B":np.array([[1., 2, 3], [4, 5, 6]]),  # n*k
            "C":np.array([[1., 2], [3, 4], [5, 6], [7, 8]]),  # j*n
            "D":np.array([[1., 2, 3], [5, 6, 7], [9, 1, 2], [2, 3, 4]]),  # j*k
            "Q":np.array([[2., 3], [4, 5]]),  # n*n
            "R":np.array([[1., 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [2, 3, 4, 5]]),  # j*j
            "state_pdf":pb.GaussPdf(np.array([1., 2]), np.array([[1., 0], [0, 2]]))  # n
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
            self.assertRaises(Exception, pb.KalmanFilter, **setup)

        # invalid dimension
        del args[6]  # remove state_pdf
        for arg in args:
            setup = self.setup_1.copy()
            setup[arg] = np.array([[1.], [2.]])
            self.assertRaises(ValueError, pb.KalmanFilter, **setup)
        gauss = pb.GaussPdf(np.array([1.]), np.array([[1.]]))
        setup = self.setup_1.copy()
        setup['state_pdf'] = gauss
        self.assertRaises(ValueError, pb.KalmanFilter, **setup)

    def test_bayes_evidence(self):
        k = pb.KalmanFilter(**self.setup_2)
        y = np.array([[4.1], [-0.2], [1.4], [-2.1]])
        u = np.array([[4.8], [-0.3], [1.1], [-1.8]])
        exp_mu = np.array([
            [ 3.62004716, -0.46320771],
            [-0.16638519,  3.58787721],
            [ 1.21108425,  0.0224309 ],
            [-1.87141692,  0.98517451]
        ])
        exp_var = np.array([
            [ 0.00999960, 40.3342872],
            [ 0.00999029,  0.20999610],
            [ 0.00963301,  0.20962422],
            [ 0.00963191,  0.20930431]
        ])
        # in:  y_t -1,  y_t,  y_t + 1
        exp_evidences_log = np.array([
            [ -3.68958564, -3.68287128,  -3.68015356],
            [ -3.16749303, -2.75744797,  -2.44453198],
            [ -2.69696927, -8.75357053, -18.48011903],
            [-10.17228316, -3.47352413,  -0.45566728]
        ])
        for i in range(4):
            k.bayes(y[i], u[i])
            post = k.posterior()
            self.assertApproxEqual(post.mean(), exp_mu[i])
            self.assertApproxEqual(post.variance(), exp_var[i])
            evidences = np.array([k.evidence_log(y[i] - 1.), k.evidence_log(y[i]), k.evidence_log(y[i] + 1.)])
            self.assertApproxEqual(evidences, exp_evidences_log[i])

    def test_copy(self):
        """Test that copying KF works as expected"""
        o = pb.KalmanFilter(**self.setup_1)  # original
        c = copy(o)  # copy
        self.assertEqual(type(c), type(o))

        self.assertNotEqual(id(o), id(c))
        self.assertArraysSame(o.A, c.A)
        self.assertArraysSame(o.B, c.B)
        self.assertArraysSame(o.C, c.C)
        self.assertArraysSame(o.D, c.D)
        self.assertArraysSame(o.Q, c.Q)
        self.assertArraysSame(o.R, c.R)
        self.assertEqual(o.n, c.n)
        self.assertEqual(o.k, c.k)
        self.assertEqual(o.j, c.j)
        self.assertEqual(id(o.P), id(c.P))
        self.assertEqual(id(o.S), id(c.S))

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
