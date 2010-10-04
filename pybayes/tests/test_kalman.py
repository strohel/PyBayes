# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for kalman"""

import unittest as ut

from numpy import array, mat

import pybayes as pb
from support import approx_eq


class TestKalman(ut.TestCase):
    """Tests for kalman filter"""

    def setUp(self):
        # synthetic parameters. May be completely mathematically invalid
        self.setup_1 = {  # n = 2, k = 3, j = 4
            "A":array([[1, 2], [3, 4]]),  # n*n
            "B":array([[1, 2, 3], [4, 5, 6]]),  # n*k
            "C":array([[1, 2], [3, 4], [5, 6], [7, 8]]),  # j*n
            "D":array([[1, 2, 3], [5, 6, 7], [9, 1, 2], [2, 3, 4]]),  # j*k
            "Q":array([[2, 3], [4, 5]]),  # n*n
            "R":array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [2, 3, 4, 5]]),  # j*j
            "state_pdf":pb.GaussPdf(array([1, 2]), array([[1, 0], [0, 2]]))  # n
        }
        self.setup_2 = {  # n = 2, k = 1, j = 1
            "A":array([[1.0, -0.5],[1.0, 0.0]]),
            "B":array([[1.0],[0.1]]),
            "C":array([[1.0, 0.0]]),
            "D":array([[0.1]]),
            "Q":array([[0.2, 0.0],[0.0, 0.2]]),
            "R":array([[0.01]]),
            "state_pdf":pb.GaussPdf(array([0.0, 0.0]), array([[200.0, 0.0],[0.0, 200.0]]))
        }

    def test_init(self):
        k = pb.Kalman(**self.setup_1)
        self.assertEqual(type(k), pb.Kalman)
        l = pb.Kalman(**self.setup_2)
        self.assertEqual(type(l), pb.Kalman)

    def test_invalid_init(self):
        args = ["A", "B", "C", "D", "Q", "R", "state_pdf"]

        # invalid type:
        for arg in args:
            setup = self.setup_1.copy()
            setup[arg] = mat([[1,2],[3,4]])
            self.assertRaises(TypeError, pb.Kalman, **setup)

        # invalid dimension
        del args[6]  # remove state_pdf
        for arg in args:
            setup = self.setup_1.copy()
            setup[arg] = array([[1],[2]])
            self.assertRaises(ValueError, pb.Kalman, **setup)
        gauss = pb.GaussPdf(array([1]), array([[1]]))
        setup = self.setup_1.copy()
        setup['state_pdf'] = gauss
        self.assertRaises(ValueError, pb.Kalman, **setup)

    def test_bayes(self):
        k = pb.Kalman(**self.setup_2)
        y = array([[4.1], [-0.2], [1.4], [-2.1]])
        u = array([[4.8], [-0.3], [1.1], [-1.8]])
        exp_x = array([
            [ 3.62004716, -0.46320771],
            [-0.16638519,  3.58787721],
            [ 1.21108425,  0.0224309 ],
            [-1.87141692,  0.98517451]
        ])
        for i in xrange(4):
            x = k.bayes(y[i], u[i])
            self.assertTrue(approx_eq(x, exp_x[i]), "Arrays {0} and {1} are not fuzzy equal."
                .format(x, exp_x[i]))
