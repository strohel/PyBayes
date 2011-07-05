# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for wrappers._numpy"""

from numpy import array, eye, dot

import pybayes.wrappers._numpy as nw
from support import PbTestCase


class TestWrappersNumpy(PbTestCase):

    def test_dot(self):
        # test data
        A = array([[1., 2.], [3., -4.]])
        B = array([[2., 1.], [3.,  2.]])
        x = array([-4., -7.])

        # precomputed restults
        AB = array([[ 8.,  5.], [-6., -5.]])
        BA = array([[ 5.,  0.], [ 9., -2.]])
        Ax = array([-18.,  16.])
        Bx = array([-15., -26.])

        # do the check!
        for (left, right, exp) in [(A, B, AB), (B, A, BA), (A, x, Ax), (B, x, Bx)]:
            res = nw.dot(left, right)
            self.assertApproxEqual(res, exp)

    def test_dot_dimensions(self):
        """Test that dot(a, b) works with different combinations of matrix dimensions"""
        A = array([[1., 2.]])
        B = array([[1., 2., 3.],
                   [4., 5., 6.]])
        self.assertApproxEqual(nw.dot(A, B), dot(A, B))  # second dot call is from NumPy
        self.assertApproxEqual(nw.dot(B.T, A.T), dot(B.T, A.T))
        with self.assertRaises(ValueError):
            nw.dot(A.T, B.T)
        with self.assertRaises(ValueError):
            nw.dot(B, A)

        C = array([[1., 2.],
                   [3., 4.],
                   [5., 6.]])
        D = array([[1.],
                   [2.]])
        self.assertApproxEqual(nw.dot(C, D), dot(C, D))
        self.assertApproxEqual(nw.dot(D.T, C.T), dot(D.T, C.T))
        with self.assertRaises(ValueError):
            nw.dot(C.T, D.T)
        with self.assertRaises(ValueError):
            nw.dot(D, C)

        # test all (transposed, smaller-dimension first) combinations
        self.assertApproxEqual(nw.dot(D.T, B), dot(D.T, B))
        self.assertApproxEqual(nw.dot(B.T, D), dot(B.T, D))
        self.assertApproxEqual(nw.dot(A, C.T), dot(A, C.T))
        self.assertApproxEqual(nw.dot(C, A.T), dot(C, A.T))
        with self.assertRaises(ValueError):
            nw.dot(D, B.T)
        with self.assertRaises(ValueError):
            nw.dot(B, D.T)
        with self.assertRaises(ValueError):
            nw.dot(A.T, C)
        with self.assertRaises(ValueError):
            nw.dot(C.T, A)

    def test_dotmv_dimensions(self):
        A = array([[1., 2.]])
        B = array([[1., 2.],
                   [3., 4.],
                   [5., 6.]])
        # while AT.T = A, its internal is dirrefent and sometimes causes problems. we try to detect them
        AT = array([[1.],
                    [2.]])
        BT = array([[1., 3., 5.],
                    [2., 4., 6.]])
        x = array([-123., -345.])

        self.assertApproxEqual(nw.dot(A, x), dot(A, x))  # second dot call is from numpy
        self.assertApproxEqual(nw.dot(B, x), dot(B, x))
        self.assertApproxEqual(nw.dot(AT.T, x), dot(AT.T, x))
        self.assertApproxEqual(nw.dot(BT.T, x), dot(BT.T, x))

    def test_dot_as_in_kalman(self):
        # a specific test for a problem that occured in KalmanFilter.bayes()
        A = array([[1., -0.5], [1., 0.]])
        R = array([[200., 0.], [0., 200.]])
        R = nw.dot(nw.dot(A, R), A.T)
        R_exp = array([[250., 200.], [200., 200.]])
        self.assertApproxEqual(R, R_exp)

    def test_dot_transposed(self):
        # transposed optimised dot made problems in the past. Test it!

        # source data
        A = array([[1., -2.], [3., 4.]])
        B = array([[0.5, 1.], [-1., 2.]])
        C = array([[-2.]])
        x = array([-7., -6.])
        y = array([-.5])

        # precomputed results
        Atx = array([-25., -10.])
        Atxt = array([-25., -10.])
        AtB = array([[-2.5, 7.], [-5., 6.]])
        ABt = array([[-1.5, -5.], [5.5, 5.]])
        AtBt = array([[3.5, 5.], [3., 10.]])
        Cty = array([1.])

        # do the test!
        for (left, right, exp) in [(A.T, x, Atx), (A.T, x.T, Atxt), (A.T, B, AtB),
                                   (A, B.T, ABt), (A.T, B.T, AtBt), (C.T, y, Cty)]:
            res = nw.dot(left, right)
            self.assertApproxEqual(res, exp)

    def test_dotvv(self):
        # source data
        a1 = array([-2.])
        b1 = array([0.5])
        a2 = array([2., 3.])
        b2 = array([-1., 5.])

        # precomputed results
        a1b1 = -1.
        a2b2 = 13.

        # the test
        for (left, right, exp) in [(a1, b1, a1b1), (a2, b2, a2b2)]:
            res = nw.dotvv(left, right)
            self.assertApproxEqual(res, exp)
