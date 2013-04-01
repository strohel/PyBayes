# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for wrappers._linalg"""

import math

import numpy as np

import pybayes.wrappers._linalg as linalg
from support import PbTestCase


def test_inv_func(self):
    """Work-around so that this function can be annotated in .pxd"""
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

def test_slogdet_func(self):
    """Work-around so that this function can be annotated in .pxd"""
    arr = np.array([[1., 2.], [-3., 4.]])
    res = math.log(linalg.det(arr))
    self.assertApproxEqual(res, 2.30258509299)

def test_cholesky_func(self):
    """Work-around so that this function can be annotated in .pxd"""
    arr = np.array([[  4.,  12., -16.],
                    [ 12.,  37., -43.],
                    [-16., -43.,  98.]])
    res = linalg.cholesky(arr)
    self.assertApproxEqual(np.dot(res, res.T), arr)


class TestWrappersLinalg(PbTestCase):

    def test_inv(self):
        test_inv_func(self)

    def test_slogdet(self):
        """Test that we emulate slogdet correctly"""
        test_slogdet_func(self)

    def test_cholesky(self):
        test_cholesky_func(self)
