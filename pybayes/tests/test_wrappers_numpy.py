# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for wrappers._numpy"""

import numpy as np

import pybayes.wrappers._numpy as nw
from support import PbTestCase


class TestWrappersNumpy(PbTestCase):

    def test_vector(self):
        v = nw.vector(13)
        self.assertEqual(v.shape[0], 13)
        self.assertTrue(type(v[0]) in (float, np.float64))

    def test_index_vector(self):
        v = nw.index_vector(13)
        self.assertEqual(v.shape[0], 13)
        self.assertTrue(type(v[0]) in (int, np.int64))

    def test_matrix(self):
        m = nw.matrix(9, 11)
        self.assertEqual((m.shape[0], m.shape[1]), (9, 11))
        self.assertTrue(type(m[0, 0]) in (float, np.float64))
