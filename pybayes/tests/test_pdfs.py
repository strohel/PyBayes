# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for pdfs"""

from math import exp, log
import unittest as ut

import numpy as np

import pybayes as pb
from support import approx_eq


class TestCpdf(ut.TestCase):
    """Test abstract class CPdf"""

    def setUp(self):
        self.cpdf = pb.CPdf()

    def test_init(self):
        self.assertEqual(type(self.cpdf), pb.CPdf)

    def test_abstract_methods(self):
        # this test may fail due to bug in cython [1] that was fixed in 0.13.1
        # [1] http://trac.cython.org/cython_trac/ticket/583
        self.assertRaises(NotImplementedError, self.cpdf.shape)
        self.assertRaises(NotImplementedError, self.cpdf.cond_shape)
        self.assertRaises(NotImplementedError, self.cpdf.cmean, np.array([0.]))
        self.assertRaises(NotImplementedError, self.cpdf.cvariance, np.array([0.]))
        self.assertRaises(NotImplementedError, self.cpdf.ceval_log, np.array([0.]), np.array([0.]))
        self.assertRaises(NotImplementedError, self.cpdf.csample, np.array([0.]))


class TestPdf(ut.TestCase):
    """Test partially abstract class Pdf"""

    def setUp(self):
        self.pdf = pb.Pdf()

    def test_init(self):
        self.assertEqual(type(self.pdf), pb.Pdf)

    def test_abstract_methods(self):
        # this test may fail due to bug in cython [1]
        # [1] http://trac.cython.org/cython_trac/ticket/583
        self.assertRaises(NotImplementedError, self.pdf.shape)
        self.assertRaises(NotImplementedError, self.pdf.mean)
        self.assertRaises(NotImplementedError, self.pdf.variance)
        self.assertRaises(NotImplementedError, self.pdf.eval_log, np.array([0.]))
        self.assertRaises(NotImplementedError, self.pdf.sample)

    def test_cond_shape(self):
        self.assertEqual(self.pdf.cond_shape(), 0)

class TestUniPdf(ut.TestCase):
    """Test uniform pdf"""

    def setUp(self):
        self.uni = pb.UniPdf(np.array([-10.]), np.array([20.]))
        (self.a, self.b) = (np.array([0., -1., 2.]), np.array([1., 1., 4.]))
        self.multiuni = pb.UniPdf(self.a, self.b)

    def test_init(self):
        self.assertEqual(type(self.uni), pb.UniPdf)
        self.assertEqual(type(self.multiuni), pb.UniPdf)

    def test_invalid_init(self):
        self.assertRaises(ValueError, pb.UniPdf, np.zeros(5), np.array([1., 2., 3., -0.01, 3.]))  # b must be > a element-wise

    def test_shape(self):
        self.assertEqual(self.uni.shape(), 1)
        self.assertEqual(self.multiuni.shape(), 3)

    def test_cond_shape(self):
        # these tests are redundant, as cond_shape() is implemented in Pdf, but wont hurt
        self.assertEqual(self.uni.cond_shape(), 0)
        self.assertEqual(self.multiuni.cond_shape(), 0)

    def test_mean(self):
        self.assertTrue(np.all(self.uni.mean() == np.array([5.])))
        self.assertTrue(np.all(self.uni.cmean(None) == np.array([5.])))  # test also cond. variant
        self.assertTrue(np.all(self.multiuni.mean() == np.array([0.5, 0., 3.])))

    def test_variance(self):
        self.assertTrue(np.all(self.uni.variance() == np.array([75.])))
        self.assertTrue(np.all(self.uni.cvariance(None) == np.array([75.])))  # cond variant
        self.assertTrue(np.all(self.multiuni.variance() == np.array([1./12., 1./3., 1./3.])))

    def test_eval_log(self):
        self.assertEqual(self.uni.eval_log(np.array([-10.1])), float('-inf'))
        self.assertEqual(self.uni.ceval_log(np.array([-10.1]), None), float('-inf'))
        self.assertEqual(self.uni.eval_log(np.array([12.547])), log(1./30.))
        self.assertEqual(self.uni.ceval_log(np.array([12.547]), None), log(1./30.))
        self.assertEqual(self.uni.eval_log(np.array([-10.1])), float('-inf'))
        self.assertEqual(self.uni.ceval_log(np.array([-10.1]), None), float('-inf'))

        (x, y, z) = (0.2, 0.8, 3.141592)  # this point belongs to multiuni's interval
        one_under = np.array([[-0.1, y, z],  # one of the values is lower
                             [x, -1.04, z],
                             [x, y, -3.975]])
        one_over = np.array([[1465.67, y, z],  # one of the values is greater
                             [x, 1.000456, z],
                             [x, y, 4.67]])
        self.assertEqual(self.multiuni.eval_log(np.array([x, y, z])), log(1./(1.*2*2)))
        for i in range(3):
            self.assertEqual(self.multiuni.eval_log(one_under[i]), float('-inf'))
            self.assertEqual(self.multiuni.eval_log(one_over[i]), float('-inf'))

    def test_sample(self):
        for i in range(100):
            sample = self.uni.sample()
            csample = self.uni.csample(None)
            self.assertTrue(-10. <= sample[0] <= 20.)  # test sample is within bounds
            self.assertTrue(-10. <= csample[0] <= 20.)  # also for conditional variant

        for i in range(100):
            sample = self.multiuni.sample()
            self.assertTrue(np.all(self.a <= sample))
            self.assertTrue(np.all(sample <= self.b))


class TestGaussPdf(ut.TestCase):
    """Test Gaussian pdf"""

    def setUp(self):
        # constructor parameters:
        self.mean = np.array([1., 3., 9.])
        self.covariance = np.array([
            [1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.]
        ])

        # expected values:
        self.variance = np.array([1., 2., 3.])  # diagonal elements of covariance
        self.shape = 3  # shape of random variable (and mean)

        self.gauss = pb.GaussPdf(self.mean, self.covariance)

    def test_init(self):
        self.assertEqual(type(self.gauss), pb.GaussPdf)

    def test_invalid_init(self):
        constructor = pb.GaussPdf

        # invalid mean and variance shape
        self.assertRaises(ValueError, constructor, np.array([[1], [2]]), self.covariance)
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

    def test_shape(self):
        self.assertEqual(self.gauss.shape(), self.shape)

    def test_mean(self):
        self.assertTrue(np.all(self.gauss.mean() == self.mean))

    def test_variance(self):
        # this test may fail due to bug in cython [1] that was fixed in 0.13.1
        # [1] http://trac.cython.org/cython_trac/ticket/583
        self.assertTrue(np.all(self.gauss.variance() == self.variance))

    def test_eval_log(self):
        x = np.array([0.])
        norm = pb.GaussPdf(np.array([0.]), np.array([[1.]]))
        expected = np.array([
            1.48671951473e-06,
            0.000133830225765,
            0.00443184841194,
            0.0539909665132,
            0.241970724519,
            0.398942280401,
            0.241970724519,
            0.0539909665132,
            0.00443184841194,
            0.000133830225765,
            1.48671951473e-06,
        ])
        for i in xrange(0, 11):
            x[0] = i - 5
            res = exp(norm.eval_log(x))
            self.assertTrue(approx_eq(res, expected[i]), "Values {0} and {1} are not fuzzy equal"
                .format(res, expected[i]))

        # non-zero mean:
        norm = pb.GaussPdf(np.array([17.9]), np.array([[1.]]))
        for i in xrange(0, 11):
            x[0] = i - 5. + 17.9
            res = exp(norm.eval_log(x))
            self.assertTrue(approx_eq(res, expected[i]), "Values {0} and {1} are not fuzzy equal"
                .format(res, expected[i]))

    def test_sample(self):
        # we cannost test values, just test right dimension and shape
        x = self.gauss.sample()
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape[0], self.mean.shape[0])

        # following test is interactive. Tester must create and check histogram:
        #norm = pb.pdfs.GaussPdf(np.array([0.]), np.array([[1.]]))
        #for i in xrange(0, 1000):
        #    print norm.sample()[0]


class TestProdPdf(ut.TestCase):
    """Test unconditional product of unconditional pdfs"""

    def setUp(self):
        self.uni = pb.UniPdf(np.array([0., 0.]), np.array([1., 2.]))
        self.gauss = pb.GaussPdf(np.array([0.]), np.array([[1.]]))
        self.prod = pb.ProdPdf(np.array([self.uni, self.gauss]))

    def test_shape(self):
        self.assertEqual(self.prod.shape(), self.uni.shape() + self.gauss.shape())

    def test_mean(self):
        mean = self.prod.mean()
        self.assertTrue(np.all(mean[0:2] == self.uni.mean()))
        self.assertTrue(np.all(mean[2:3] == self.gauss.mean()))

    def test_variance(self):
        variance = self.prod.variance()
        self.assertTrue(np.all(variance[0:2] == self.uni.variance()))
        self.assertTrue(np.all(variance[2:3] == self.gauss.variance()))

    def test_eval_log(self):
        test_points = np.array([  # point we evaluate product in
            [-1., -1., 0.],
            [0.4, -1., 0.],
            [0.4, 1.7, 0.],
            [0.4, 2.1, 0.],
            [1.2, 2.1, 0.]
        ])
        p = exp(self.gauss.eval_log(np.array([0.]))) # = 1/sqrt(2*PI)
        exp_values = np.array([  # expected values. (not the logarithm of them)
            0.*p,  # 0*0*p
            0.*p,  # 1*0*p
            0.5*p,  # 1*0.5*p
            0.*p,  # 1*0*p
            0.*p,  # 0*0*p
        ])
        for i in range(test_points.shape[0]):
            val = exp(self.prod.eval_log(test_points[i]))
            self.assertEqual(val, exp_values[i])

    def test_sample(self):
        # we can only somehow test unifrom pdf, so we create a product of them
        uni_list = []
        for i in range(10):
            uni_list.append(pb.UniPdf(np.array([i+0.]), np.array([i+1.])))
        uni_prod = pb.ProdPdf(np.array(uni_list))  # a product of 10 UniPdfs
        for i in range(100):
            sample = uni_prod.sample()
            for j in range(10): # test each component..
                self.assertTrue(j <= sample[j] <= j + 1)  # ..is within bounds


class TestMLinGaussCPdf(ut.TestCase):
    """Test conditional Gaussian pdf with mean as a linear function of cond"""

    def setUp(self):
        # constructor parameters:
        self.A = np.array([
            [1., 0.],
            [0., 2.],
            [-1., -1.]
        ])
        self.b = np.array([-0.5, -1., -1.5])
        self.covariance = np.array([
            [1., 0., 0.],
            [0., 2., 0.],
            [0., 0., 3.]
        ])

        # expected values:
        self.variance = np.array([1., 2., 3.])  # diagonal elements of covariance
        self.shape = 3  # shape of random variable (and mean)
        self.cond_shape = 2

        self.test_conds = np.array([  # array of test conditions (shared by various tests)
            [0., 0.],
            [1., 0.],
            [0., -1.],
        ])
        self.cond_means = np.array([  # array of mean values that match (first n entries of) test_conds
            self.b,
            np.array([1., 0., -1.]) + self.b,  # computed from self.A
            np.array([0., -2., 1.]) + self.b,  # computed from self.A
        ])

        self.gauss = pb.MLinGaussCPdf(self.covariance, self.A, self.b)

    def test_init(self):
        self.assertEqual(type(self.gauss), pb.MLinGaussCPdf)

    def test_invalid_init(self):
        constructor = pb.MLinGaussCPdf

        # covariance not compatible with A, b:
        self.assertRaises(ValueError, constructor, np.array([[1.,0.],[0.,1.]]), self.A, self.b)
        # A not compatible with covariance:
        self.assertRaises(ValueError, constructor, self.covariance, np.array([[1.,0.],[0.,1.]]), self.b)
        # b not compatible with covariance:
        self.assertRaises(ValueError, constructor, self.covariance, self.A, np.array([1., 2.]))

    def test_shape(self):
        self.assertEqual(self.gauss.shape(), self.shape)

    def test_cond_shape(self):
        self.assertEqual(self.gauss.cond_shape(), self.cond_shape)

    def test_cmean(self):
        for i in range(self.cond_means.shape[0]):
            cmean = self.gauss.cmean(self.test_conds[i])
            self.assertTrue(np.all(cmean == self.cond_means[i]))

    def test_cvariance(self):
        for i in range(self.test_conds.shape[0]):
            # test all test conditions, the result must not depend on it
            self.assertTrue(np.all(self.gauss.cvariance(self.test_conds[i]) == self.variance))

    def test_ceval_log(self):
        x = np.array([0.])
        cond = np.array([0.])
        norm = pb.MLinGaussCPdf(np.array([[1.]]), np.array([[1.]]), np.array([-1.]))
        expected = np.array([
            1.48671951473e-06,
            0.000133830225765,
            0.00443184841194,
            0.0539909665132,
            0.241970724519,
            0.398942280401,
            0.241970724519,
            0.0539909665132,
            0.00443184841194,
            0.000133830225765,
            1.48671951473e-06,
        ])
        for i in xrange(0, 11):
            # cond is set to [1.], which should produce mean = [0.]
            x[0] = i - 5.
            cond[0] = 1.
            res = exp(norm.ceval_log(x, cond))
            self.assertTrue(approx_eq(res, expected[i]), "Values {0} and {1} are not fuzzy equal"
                .format(res, expected[i]))

            # cond is set to [456.78], which should produce mean = [455.78]
            x[0] = i - 5. + 455.78
            cond[0] = 456.78
            res = exp(norm.ceval_log(x, cond))
            self.assertTrue(approx_eq(res, expected[i]), "Values {0} and {1} are not fuzzy equal"
                .format(res, expected[i]))

    def test_csample(self):
        # we cannost test values, just test right dimension and shape
        for i in range(self.test_conds.shape[0]):
            x = self.gauss.csample(self.test_conds[i])
            self.assertEqual(x.ndim, 1)
            self.assertEqual(x.shape[0], self.covariance.shape[0])
