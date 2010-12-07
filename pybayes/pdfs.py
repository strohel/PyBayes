# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
This module contains models of common probability density functions, abbreviated
as pdfs.

All classes from this module are currently imported to top-level pybayes module,
so instead of ``from pybayes.pdfs import Pdf`` you can type ``from pybayes import
Pdf``.
"""

from math import log

from numpywrap import *


class CPdf(object):
    """Base class for all Conditional Probability Density Functions

    When you evaluate a CPdf the result also depends on condition (vector), in
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
        """Return True if cond has correct type and shape, raise Error otherwise"""
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
    r"""Simple uniform multivariate probability density function

    .. math:: f(x) = \Theta(x - a) \Theta(b - x) \prod_{i=1}^n \frac{1}{b_i-a_i}
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
    r"""Unconditional Gaussian (normal) probability density function

    .. math:: f(x) \propto \exp \left( - \left( x-\mu \right)' R^{-1} \left( x-\mu \right) \right)
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


class ProdPdf(Pdf):
    r"""Unconditional product of multiple unconditional pdfs.

    You can for example create a pdf that has uniform distribution with regards
    to x-axis and normal distribution along y-axis. The caller (you) must ensure
    that individial random variables are independent, otherwise their product may
    have no mathematical sense.

    .. math:: f(x_1 x_2 x_3) = f_1(x_1) f_2(x_2) f_3(x_3)
    """

    def __init__(self, factors):
        """Construct product of unconditional pdfs.

        .. factors: nunpy.ndarray whose elements are Pdf objects
        >>> prod = ProdPdf(numpy.array([UniPdf(...), GaussPdf(...)]))
        """
        self.factors = asarray(factors)
        if self.factors.ndim != 1:
            raise ValueError("factors must be 1D numpy.ndarray")
        self.shapes = zeros(self.factors.shape[0], dtype=int)  # array of factor shapes
        for i in range(self.factors.shape[0]):
            if not isinstance(self.factors[i], Pdf):
                raise TypeError("all records in factors must be (subclasses of) Pdf")
            self.shapes[i] = self.factors[i].shape()

        # pre-calclate shape
        self._shape = sum(self.shapes)

    def shape(self):
        return self._shape

    def mean(self):
        curr = 0
        ret = zeros(self.shape())
        for i in range(self.factors.shape[0]):
            ret[curr:curr + self.shapes[i]] = self.factors[i].mean()
            curr += self.shapes[i]
        return ret;

    def variance(self):
        curr = 0
        ret = zeros(self.shape())
        for i in range(self.factors.shape[0]):
            ret[curr:curr + self.shapes[i]] = self.factors[i].variance()
            curr += self.shapes[i]
        return ret;

    def eval_log(self, x):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")
        curr = 0
        ret = 0.  # 1 is neutral element in multiplication; log(1) = 0
        for i in range(self.factors.shape[0]):
            ret += self.factors[i].eval_log(x[curr:curr + self.shapes[i]])  # log(x*y) = log(x) + log(y)
            curr += self.shapes[i]
        return ret;

    def sample(self):
        curr = 0
        ret = zeros(self.shape())
        for i in range(self.factors.shape[0]):
            ret[curr:curr + self.shapes[i]] = self.factors[i].sample()
            curr += self.shapes[i]
        return ret;


class MLinGaussCPdf(CPdf):
    r"""Conditional Gaussian pdf whose mean is a linear function of condition

    .. math::

       f(x|c) \propto \exp \left( - \left( x-\mu \right)' R^{-1} \left( x-\mu \right) \right)
       \quad \quad \text{where} ~ \mu := A c + b
    """

    def __init__(self, covariance, A, b):
        """Initialise Mean-Linear Gaussian conditional pdf.

        covariance - covariance of underlying Gaussian pdf
        A, b: given condition cond, mean = A*cond + b
        """
        self.gauss = GaussPdf(zeros(covariance.shape[0]), covariance)
        self.A = asarray(A)
        self.b = asarray(b)
        if self.A.ndim != 2:
            raise ValueError("A must be 2D numpy.ndarray (matrix)")
        if self.b.ndim != 1:
            raise ValueError("b must be 1D numpy.ndarray (vector)")
        if self.b.shape[0] != self.gauss.shape():
            raise ValueError("b must have same number of cols as covariance")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A must have same number of rows as covariance")

    def shape(self):
        return self.b.shape[0]

    def cond_shape(self):
        return self.A.shape[1]

    def cmean(self, cond):
        self.check_cond(cond)
        return dot(self.A, cond) + self.b

    def cvariance(self, cond):
        # cond is unused here, so no need to check it
        return self.gauss.variance()

    def ceval_log(self, x, cond):
        # cond is checked in cmean()
        self.gauss.mu = self.cmean(cond)
        return self.gauss.eval_log(x)

    def csample(self, cond):
        # cond is checked in cmean()
        self.gauss.mu = self.cmean(cond)
        return self.gauss.sample()
