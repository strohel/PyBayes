# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for wrappers._linalg"""

import numpy as np

import pybayes.wrappers._linalg as linalg
from support import PbTestCase


class TestWrappersLinalg(PbTestCase):

    def test_inv(self):
        # source data
        arrays = [
            np.array([[ 2.]]),
            np.array([[ 0.,  2.], [ 3.,  0.]]),
            np.array([[ 1., -2.], [-4.,  9.]]),
            np.array([[10., 11.], [100., 111.]]),  # near singular
            np.array([[1., 2., -3.], [1., -2., 3.], [-1., 2., 3.]])
        ]

        # test that A * inv(A) = I within machine precision
        for A in arrays:
            iA = linalg.inv(A)
            E = np.eye(A.shape[0])
            E1 = np.dot(A, iA)
            E2 = np.dot(iA, A)
            self.assertApproxEqual(E1, E)
            self.assertApproxEqual(E2, E)

    def test_slogdet(self):
        """Test that we have defined slogdet the correct way for older NumPy versions"""
        arr = np.array([[1., 2.], [3., 4.]])
        res = linalg.slogdet(arr)
        self.assertEqual(res, (-1., 0.6931471805599454))
