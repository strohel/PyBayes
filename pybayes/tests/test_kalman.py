# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for kalman"""

import unittest as ut

from numpy import array, mat

import pybayes as pb

class TestKalman(ut.TestCase):
    """Tests for kalman filter"""

    def setUp(self):
        # synthetic parameters. May be completely mathematically invalid
        self.setup_1 = { # n = 2, k = 3, j = 4
            "A":array([[1, 2], [3, 4]]),  # n*n
            "B":array([[1, 2, 3], [4, 5, 6]]),  # n*k
            "C":array([[1, 2], [3, 4], [5, 6], [7, 8]]),  # j*n
            "D":array([[1, 2, 3], [5, 6, 7], [9, 1, 2], [2, 3, 4]]),  # j*k
            "Q":array([[2, 3], [4, 5]]),  # n*n
            "R":array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [2, 3, 4, 5]]),  # n*n
            "state_pdf":pb.GaussPdf(array([1, 2]), array([[1, 0], [0, 2]]))  # n
        }

    def test_init(self):
        k = pb.Kalman(**self.setup_1)
        self.assertEqual(type(k), pb.Kalman)

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
        # TODO
        pass
