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


class RVComp(object):
    """Atomic component of a random variable."""

    def __init__(self, name, dimension):
        """Initialise new component of a random variable :class:`RV`.

        :param name: name of the component. Pass None for an anonymous component
        :type name: string or None
        :param dimension: number of vector components this component occupies
        :type dimension: positive integer
        :raises TypeError: non-integer dimension or non-string name
        :raises ValueError: invalid dimension
        """

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be either None or a string")
        self.name = name
        if not isinstance(dimension, int):
            raise TypeError("dimension must be integer (int)")
        if dimension < 1:
            raise ValueError("dimension must be non-zero positive")
        self.dimension = dimension


class RV(object):
    """Representation of a random variable made of one or more components. See
    :class:`RVComp`"""

    def __init__(self, *components):
        """Initialise new random variable.

        :param \*components: components that should form the random vector. You may
            also pass another RVs which is a shotrcut for adding all their components.
        :type \*components: :class:`RV` or :class:`RVComp`
        :raises TypeError: invalid object passed (neither a :class:`RV` or a :class:`RVComp`)
        :raises ValueError: zero components passed

        Usual way of creating RV could be:

        >>> x = RV(RVComp('x_1', 1), RVComp('x_2', 1))
        >>> x.name
        '[x_1, x_2]'
        >>> xy = RV(x, RVComp('y', 2))
        >>> xy.name
        '[x_1, x_2, y]'
        """
        self.dimension = 0
        self.name = '['
        self.components = []
        if len(components) is 0:
            raise ValueError("at least one component must be passed")
        for component in components:
            if isinstance(component, RVComp):
                self._add_component(component)
            elif isinstance(component, RV):
                for subcomp in component.components:
                    self._add_component(subcomp)
            else:
                raise TypeError('component ' + component + ' is neither an instance '
                              + 'of RVComp or RV')
        self.name = self.name[:-2] + ']'

    def _add_component(self, component):
        """Add new component to this random variable.

        Internal function, do not use outside of PyBayes"""
        # TODO: check if component is already contained? (does it matter somewhere?)
        self.components.append(component)
        self.dimension += component.dimension
        self.name += component.name + ", "

    def contains(self, component):
        """Return True if this random variable contains the exact same instance of
        the component

        :param component: component whose presence you want to test
        :type component: :class:`RVComp`
        :rtype: bool
        """
        for comp in self.components:
            if id(comp) == id(component):
                return True
        return False


class CPdf(object):
    """Base class for all Conditional Probability Density Functions.

    When you evaluate a CPdf the result generally also depends on a condition
    (vector) named `cond` in PyBayes. For a CPdf that is a :class:`Pdf` this is
    not the case, the result is unconditional.
    """

    def shape(self):
        """Return shape of the random variable (and mean).

        :rtype: int"""
        raise NotImplementedError("Derived classes must implement this function")

    def cond_shape(self):
        """Return shape of the condition.

        :rtype: int"""
        raise NotImplementedError("Derived classes must implement this function")

    def mean(self, cond = None):
        """Return (conditional) mean value of the pdf.

        :rtype: numpy.ndarray"""
        raise NotImplementedError("Derived classes must implement this function")

    def variance(self, cond = None):
        """Return (conditional) variance (diagonal elements of covariance).

        :rtype: numpy.ndarray"""
        raise NotImplementedError("Derived classes must implement this function")

    def eval_log(self, x, cond = None):
        """Return logarithm of (conditional) likelihood function in point x.

        :param x: point which to evaluate the function in
        :type x: numpy.ndarray
        :rtype: double"""
        raise NotImplementedError("Derived classes must implement this function")

    def sample(self, cond = None):
        """Return one random (conditional) sample from this distribution

        :rtype: numpy.ndarray"""
        raise NotImplementedError("Derived classes must implement this function")

    def check_cond(self, cond):
        """Return True if cond has correct type and shape, raise Error otherwise

        :raises TypeError: cond is not of correct type
        :raises ValueError: cond doesn't have appropriate shape
        :rtype: bool"""
        if cond is None:  # cython-specific
            raise TypeError("cond must be numpy.ndarray")
        if cond.ndim != 1:
            raise ValueError("cond must be 1D numpy array (a vector)")
        if cond.shape[0] != self.cond_shape():
            raise ValueError("cond must be of shape ({0},) array of shape ({1},) given".format(cond_shape(), cond.shape[0]))
        return True


class Pdf(CPdf):
    """Base class for all unconditional (static) multivariate Probability Density
    Functions. Subclass of CPdf."""

    def cond_shape(self):
        """Return zero as Pdfs have no condition."""
        return 0


class UniPdf(Pdf):
    r"""Simple uniform multivariate probability density function.

    .. math:: f(x) = \Theta(x - a) \Theta(b - x) \prod_{i=1}^n \frac{1}{b_i-a_i}

    :var numpy.ndarray a: left border
    :var numpy.ndarray b: right border

    You may modify these attributes as long as you don't change their shape and
    assumption **a** < **b** still holds.
    """

    def __init__(self, a, b):
        """Initialise uniform distribution.

        :param a: left border
        :type a: numpy.ndarray
        :param b: right border
        :type b: numpy.ndarray

        **b** must be greater (in each dimension) than **a**
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

    def mean(self, cond = None):
        return (self.a+self.b)/2.  # element-wise division

    def variance(self, cond = None):
        return ((self.b-self.a)**2)/12.  # element-wise power and division

    def eval_log(self, x, cond = None):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")
        if np_any(x <= self.a) or np_any(x >= self.b):
            return float('-inf')
        return -log(prod(self.b-self.a))

    def sample(self, cond = None):
        return uniform(-0.5, 0.5, self.shape()) * (self.b-self.a) + self.mean()


class GaussPdf(Pdf):
    r"""Unconditional Gaussian (normal) probability density function.

    .. math:: f(x) \propto \exp \left( - \left( x-\mu \right)' R^{-1} \left( x-\mu \right) \right)

    :var numpy.ndarray mu: mean value
    :var numpy.ndarray R: covariance matrix

    You can modify object parameters only if you are absolutely sure that you
    pass allowable values, because parameters are only checked once in constructor.
    """

    def __init__(self, mean=array([0]), covariance=array([[1]])):
        """Initialise Gaussian pdf.

        :param numpy.ndarray mean: mean value (1D array)
        :param numpy.ndarray covariance: covariance matrix (2D array)
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

    def mean(self, cond = None):
        return self.mu

    def variance(self, cond = None):
        return diag(self.R)

    def eval_log(self, x, cond = None):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")

        # compute logarithm of normalization constant (can be cached in future)
        # log(2*Pi) = 1.83787706640935
        # we ignore sign (first part of slogdet return value) as it must be positive
        log_norm = -1/2. * (self.mu.shape[0]*1.83787706640935 + slogdet(self.R)[1])

        # part that actually depends on x
        log_val = -1/2. * dotvv(x - self.mu, dot(inv(self.R), x - self.mu))
        return log_norm + log_val  # = log(norm*val)

    def sample(self, cond = None):
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

        :param factors: array whose elements are Pdf objects
        :type factors: numpy.ndarray

        Usual way of creating ProdPdf could be:

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

    def mean(self, cond = None):
        curr = 0
        ret = zeros(self.shape())
        for i in range(self.factors.shape[0]):
            ret[curr:curr + self.shapes[i]] = self.factors[i].mean()
            curr += self.shapes[i]
        return ret;

    def variance(self, cond = None):
        curr = 0
        ret = zeros(self.shape())
        for i in range(self.factors.shape[0]):
            ret[curr:curr + self.shapes[i]] = self.factors[i].variance()
            curr += self.shapes[i]
        return ret;

    def eval_log(self, x, cond = None):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")
        curr = 0
        ret = 0.  # 1 is neutral element in multiplication; log(1) = 0
        for i in range(self.factors.shape[0]):
            ret += self.factors[i].eval_log(x[curr:curr + self.shapes[i]])  # log(x*y) = log(x) + log(y)
            curr += self.shapes[i]
        return ret;

    def sample(self, cond = None):
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

    def mean(self, cond = None):
        self.check_cond(cond)
        return dot(self.A, cond) + self.b

    def variance(self, cond = None):
        # cond is unused here, so no need to check it
        return self.gauss.variance()

    def eval_log(self, x, cond = None):
        # cond is checked in mean()
        self.gauss.mu = self.mean(cond)
        return self.gauss.eval_log(x)

    def sample(self, cond = None):
        # cond is checked in mean()
        self.gauss.mu = self.mean(cond)
        return self.gauss.sample()


class ProdCPdf(CPdf):
    r"""Pdf that is formed as a chain rule of multiple conditional pdfs.

    .. math:: f(x_1 x_2 x_3 | c) = f_1(x_1 | x_2 x_3 c) f_2(x_2 | x_3 c) f_3(x_3 | c)
    """

    def __init__(self, factors):
        """Construct chain rule of multiple cpdfs.

        .. factors - 1D numpy.ndarray of objects of type CPdf (or subclasses)
        """
        self.factors = asarray(factors)
        if self.factors.ndim != 1:
            raise ValueError("factors must be 1D numpy.ndarray")

        self.shapes = zeros(self.factors.shape[0], dtype=int)  # array of factor shapes

        accumulate_cond_shape = 0
        for i in range(self.factors.shape[0] -1, -1, -1):
            if not isinstance(self.factors[i], CPdf):
                raise TypeError("all records in factors must be (subclasses of) CPdf")
            self.shapes[i] = self.factors[i].shape()
            if self.shapes[i] == 0:
                raise ValueError("ProdCPdf cannot contain zero-shaped factors (factor {0})".format(i))
            if accumulate_cond_shape == 0:  # the last factor
                self._cond_shape = self.factors[i].cond_shape()
                accumulate_cond_shape += self._cond_shape
            else:  # other factors
                if self.factors[i].cond_shape() != accumulate_cond_shape:
                    raise ValueError("Expected cond_shape() of factor {0} will be {1}, ".format(i, accumulate_cond_shape)
                              + "got {0}. (because factor on the right has ".format(self.factors[i].cond_shape())
                              + "shape() {0} and cond_shape() {1}".format(self.shapes[i+1], self.factors[i+1].cond_shape()))

            # prepare for next iteration:
            accumulate_cond_shape += self.shapes[i]

        # pre-calculate shape
        self._shape = sum(self.shapes)

    def shape(self):
        return self._shape

    def cond_shape(self):
        return self._cond_shape

    def mean(self, cond = None):
        raise NotImplementedError("Not yet implemented")

    def variance(self, cond = None):
        raise NotImplementedError("Not yet implemented")

    def eval_log(self, x, cond = None):
        if x is None:  # cython-specific, but wont hurt in python
            raise TypeError("x must be numpy.ndarray")
        self.check_cond(cond)

        start = 0
        cond_start = 0

        ret = 0.
        comb_input = zeros(self.shape() + self.cond_shape())  # combined x and cond
        comb_input[:self.shape()] = x  # TODO: check that x has right shape
        comb_input[self.shape():] = cond

        for i in range(self.factors.shape[0]):
            cond_start += self.shapes[i]
            ret += self.factors[i].eval_log(comb_input[start:cond_start], comb_input[cond_start:])
            start += self.shapes[i]
        return ret

    def sample(self, cond = None):
        self.check_cond(cond)

        # combination of return value and condition
        comb = zeros(self.shape() + self.cond_shape())
        start = self.shape()
        comb[start:] = cond

        for i in range(self.factors.shape[0] -1, -1, -1):
            stop = start
            start -= self.shapes[i]
            comb[start:stop] = self.factors[i].sample(comb[stop:])

        return comb[:self.shape()]
