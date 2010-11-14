# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Probability density functions"""

from math import log

from numpywrap import *


class CPdf(object):
    """Base class for all Conditional Probability Density Functions

    when you evaluate a CPdf the result also depends on condition (vector), in
    PyBayes named cond.
    """

    def shape(self):
        """Return shape of the random variable (and mean) as int"""
        raise NotImplementedError("Derived classes must implement this function")

    def cond_shape(self):
        """Return shape of the condition as int"""
        raise NotImplementedError("Derived classes must implement this function")

    def cmean(self, cond):
        """Return conditional mean value (a vector) of the pdf"""
        raise NotImplementedError("Derived classes must implement this function")

    def cvariance(self, cond):
        """Return conditional variance (diagonal elements of covariance)"""
        raise NotImplementedError("Derived classes must implement this function")

    def ceval_log(self, x, cond):
        """Return logarithm of conditional likelihood function in point x"""
        raise NotImplementedError("Derived classes must implement this function")

    def csample(self, cond):
        """Return one random conditional sample. Density of samples should adhere to this density"""
        raise NotImplementedError("Derived classes must implement this function")

    def check_cond(self, cond):
        """Return True id cond has correct type and shape, raise Error otherwise"""
        if cond is None:  # cython-specific
            raise TypeError("cond must be numpy.ndarray")
        if cond.ndim != 1:
            raise ValueError("cond must be 1D numpy array (a vector)")
        if cond.shape[0] != self.cond_shape():
            raise ValueError("cond must be of shape ({0},) array of shape ({1},) given".format(cond_shape(), cond.shape[0]))
        return True


class Pdf(CPdf):
    """Base class for all unconditional (static) multivariate Probability Density Functions"""

    def cond_shape(self):
        """Return shape of the condition, which is zero for unconditional Pdfs"""
        return 0

    def cmean(self, cond):
        return self.mean()

    def mean(self):
        """Return mean value (a vector) of the pdf"""
        raise NotImplementedError("Derived classes must implement this function")

    def cvariance(self, cond):
        return self.variance()

    def variance(self):
        """Return variance (diagonal elements of covariance)"""
        raise NotImplementedError("Derived classes must implement this function")

    def ceval_log(self, x, cond):
        return self.eval_log(x)

    def eval_log(self, x):
        """Return logarithm of likelihood function in point x"""
        raise NotImplementedError("Derived classes must implement this function")

    def csample(self, cond):
        return self.sample()

    def sample(self):
        """Return one random sample. Density of samples should adhere to this density"""
        raise NotImplementedError("Derived classes must implement this function")


class UniPdf(Pdf):
    """Simple uniform multivariate probability density function

    .. math: f(x|a, b) = \THETA {x-a} \THETA {b-x} {1} \over {b-a}  TODO
    """

    def __init__(self, a, b):
        """Initialise uniform distribution with left point a and right point b

        a must be greater (in each dimension) than b
        """
        self.a = asarray(a)
        self.b = asarray(b)
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("both a and b must be 1D numpy arrays (vectors)")
        if a.shape[0] != b.shape[0]:
            raise ValueError("a must have same shape as b")
        if np_any(self.b <= self.a):
            raise ValueError("b must be greater than a in each dimension")

    def shape(self):
        return self.a.shape[0]

    def mean(self):
        return (self.a+self.b)/2.  # element-wise division

    def variance(self):
        return ((self.b-self.a)**2)/12.  # element-wise power and division

    def eval_log(self, x):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")
        if np_any(x <= self.a) or np_any(x >= self.b):
            return float('-inf')
        return -log(prod(self.b-self.a))

    def sample(self):
        return uniform(-0.5, 0.5, self.shape()) * (self.b-self.a) + self.mean()


class GaussPdf(Pdf):
    """Unconditional Gaussian (normal) probability density function

    .. math: f(x|\mu, R) \propto \exp(-(x-\mu)'R^{-1}(x-\mu))
    """

    def __init__(self, mean=array([0]), covariance=array([[1]])):
        """Initialise Gaussian pdf with mean value mean and covariance matrix covariance

        mean should be 1D array and covariance must be a matrix (2D array). To make
        1D GaussPdf, pass [[number]] as covariance.

        mean is stored in mu attribute
        covariance is stored in R attribute
        you can modify object parameters only if you are absolutely sure that you
        pass correct values, because parameters are only checked once in constructor
        """
        mean = asarray(mean)
        covariance = asarray(covariance)
        if mean.ndim != 1:
            raise ValueError("mean must be one-dimensional (" + str(mean.ndim) + " dimensions encountered)")
        n = mean.shape[0]
        if covariance.shape != (n, n):
            raise ValueError("covariance must have shape (" + str(n) + ", " + str(n) + "), " +
                             str(covariance.shape) + " given")
        if np_any(covariance != covariance.T):
            raise ValueError("covariance must be symmetric (complex covariance not supported)")
        # TODO: covariance must be positive definite
        self.mu = mean
        self.R = covariance

    def shape(self):
        return self.mu.shape[0]

    def mean(self):
        return self.mu

    def variance(self):
        return diag(self.R)

    def eval_log(self, x):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")

        # compute logarithm of normalization constant (can be cached in future)
        # log(2*Pi) = 1.83787706640935
        # we ignore sign (first part of slogdet return value) as it must be positive
        log_norm = -1/2. * (self.mu.shape[0]*1.83787706640935 + slogdet(self.R)[1])

        # part that actually depends on x
        log_val = -1/2. * dotvv(x - self.mu, dot(inv(self.R), x - self.mu))
        return log_norm + log_val  # = log(norm*val)

    def sample(self):
        z = normal(size=self.mu.shape[0]);
        # NumPy's cholesky(R) is equivalent to Matlab's chol(R).transpose()
        return self.mu + dot(cholesky(self.R), z);
