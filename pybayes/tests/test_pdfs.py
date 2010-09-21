# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for pdfs"""

import unittest as ut

import numpy as np

import pybayes as pb

class TestPdf(ut.TestCase):
    """Test abstract class Pdf"""

    def setUp(self):
        self.pdf = pb.Pdf()

    def test_abstract_methods(self):
        self.assertRaises(NotImplementedError, self.pdf.mean)
        self.assertRaises(NotImplementedError, self.pdf.variance)
        self.assertRaises(NotImplementedError, self.pdf.eval_log, 0.)
        self.assertRaises(NotImplementedError, self.pdf.sample)


class TestGaussPdf(ut.TestCase):
    """Test Gaussian pdf"""

    def setUp(self):
        self.mean = np.array([1., 3., 9.])
        self.variance = np.array([
            [1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.]
        ])
        self.variance_diag = np.array([1., 2., 3.])
        self.gauss = pb.GaussPdf(self.mean, self.variance)

    def test_invalid_initialisation(self):
        constructor = pb.GaussPdf

        # invalid mean and variance shape
        self.assertRaises(ValueError, constructor, np.array([[1], [2]]), self.variance)
        self.assertRaises(ValueError, constructor, self.mean, np.array([1., 2., 3.,]))

        # non-rectangular variance, incompatible mean and variance, non-symmetric variance
        self.assertRaises(ValueError, constructor, self.mean, np.array([
            [1., 2.],
            [3., 4.],
            [5., 6.]
        ]))
        self.assertRaises(ValueError, constructor, self.mean, np.array([
            [1., 2.],
            [3., 4.]
        ]))
        self.assertRaises(ValueError, constructor, np.array([1, 2]), np.array([
            [1, 2],
            [3, 4]
        ]))

        # TODO: non positive-definite variance

    def test_mean(self):
        self.assertTrue(np.all(self.gauss.mean() == self.mean))

    def test_variance(self):
        self.assertTrue(np.all(self.gauss.variance() == self.variance_diag))  # TODO: vector or matrix?

    #def test_eval_log(self):  # TODO
        #x = [
            #self.mean,
            #np.array([0., 0., 0.])
        #]
        #exp = [
            #np.array([1, 2, 3]),  # TODO
            #np.array([1, 2, 3])  # TODO
        #]
        #for i in xrange(len(x)):
            #print "GaussPdf.eval_log(" + str(x[i]) + ") =", self.gauss.eval_log(x[i]), "(expected", str(exp[i]) + ")"

    def test_sample(self):
        # we cannost test values, just test right dimension and shape
        x = self.gauss.sample()
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape[0], self.mean.shape[0])

        # single dimension
        #normal = pb.pdfs.GaussPdf(np.array([0.]), np.array([[1.]]))
        #values = []
        #for i in xrange(0, 100):
            #values.extend(normal.sample())
        #print values
