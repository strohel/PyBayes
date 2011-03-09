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
    """Atomic component of a random variable.

    :var int dimension: dimension; do not change unless you know what you are doing
    :var str name: name; can be changed as long as it remains a string (warning:
       parent RVs are not updated)
    """

    def __init__(self, dimension, name = None):
        """Initialise new component of a random variable :class:`RV`.

        :param dimension: number of vector components this component occupies
        :type dimension: positive integer
        :param name: name of the component; default: None for anonymous component
        :type name: string or None
        :raises TypeError: non-integer dimension or non-string name
        :raises ValueError: non-positive dimension
        """

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be either None or a string")
        self.name = name
        if not isinstance(dimension, int):
            raise TypeError("dimension must be integer (int)")
        if dimension < 1:
            raise ValueError("dimension must be non-zero positive")
        self.dimension = dimension

    #def __eq__(self, other):
        #"""We want RVComp have to be hashable
        #(http://docs.python.org/glossary.html#term-hashable), but default __eq__()
        #and __hash__() implementations suffice, as they are instance-based.
        #"""


class RV(object):
    """Representation of a random variable made of one or more components. See
    :class:`RVComp`.

    :var int dimension: cummulative dimension; do not change
    :var str name: pretty name, can be changed but needs to be a string
    :var list components: list of RVComps; do not change

    *Please take into account that all RVComp comparisons inside RV are
    instance-based and component names are purely informational. To demonstrate:*

    >>> rv = RV(RVComp(1, "a"))
    >>> ...
    >>> rv.contains(RVComp(1, "a"))
    False

    Right way to do this would be:

    >>> a = RVComp(1, "arbitrary pretty name for a")
    >>> rv = RV(a)
    >>> ...
    >>> rv.contains(a)
    True
    """

    def __init__(self, *components):
        """Initialise new random variable.

        :param \*components: components that should form the random vector. You may
            also pass another RVs which is a shotrcut for adding all their components.
        :type \*components: :class:`RV`, :class:`RVComp` or a sequence of :class:`RVComp` items
        :raises TypeError: invalid object passed (neither a :class:`RV` or a :class:`RVComp`)

        Usual way of creating RV could be:

        >>> x = RV(RVComp(1, 'x_1'), RVComp(1, 'x_2'))
        >>> x.name
        '[x_1, x_2]'
        >>> xy = RV(x, RVComp(2, 'y'))
        >>> xy.name
        '[x_1, x_2, y]'
        """
        self.dimension = 0
        self.components = []
        if len(components) is 0:
            self.name = '[]'
            return

        self.name = '['
        for component in components:
            if isinstance(component, RVComp):
                self._add_component(component)
            elif isinstance(component, RV):
                for subcomp in component.components:
                    self._add_component(subcomp)
            else:
                try:
                    for subcomp in component:
                        self._add_component(subcomp)
                except TypeError:
                    raise TypeError('component ' + str(component) + ' is neither an instance '
                                + 'of RVComp or RV and is not iterable of RVComps')
        self.name = self.name[:-2] + ']'

    def _add_component(self, component):
        """Add new component to this random variable.

        Internal function, do not use outside of RV."""
        if not isinstance(component, RVComp):
            raise TypeError("component is not of type RVComp")
        self.components.append(component)
        self.dimension += component.dimension
        self.name = '{0}{1}, '.format(self.name, component.name)
        return True

    def contains(self, component):
        """Return True if this random variable contains the exact same instance of
        the component

        :param component: component whose presence is tested
        :type component: :class:`RVComp`
        :rtype: bool
        """
        return component in self.components

    def contains_all(self, test_components):
        """Return True if this RV contains all RVComps from sequence
        **components**.

        :param components: list of components whose presence is checked
        :type components: sequence of :class:`RVComp` items
        """
        for test_comp in test_components:
            if not self.contains(test_comp):
                return False
        return True;

    def contained_in(self, test_components):
        """Return True if sequence **components** contains all all components
        from this RV (and perhaps more).

        :param components: set of components whose presence is checked
        :type components: sequence of :class:`RVComp` items
        """
        for component in self.components:
            if component not in test_components:
                return False
        return True

    def indexed_in(self, super_rv):
        """Return index array such that this rv is indexed in **super_rv**, which
        must be superset of this rv. Resulting array can be used with :func:`numpy.take`
        and :func:`numpy.put`.

        :param super_rv: returned indices apply to this rv
        :type super_rv: :class:`RV`
        :rtype: 1-D :class:`numpy.ndarray` of ints with dimension = self.dimension
        """
        ret = empty(self.dimension, dtype=int)
        ret_ind = 0  # current index in returned index array
        # process each component from target rv
        for comp in self.components:
            # find associated component in source_rv components:
            src_ind = 0  # index in source vector
            for source_comp in super_rv.components:
                if source_comp is comp:
                    ret[ret_ind:] = arange(src_ind, src_ind + comp.dimension)
                    ret_ind += comp.dimension
                    break;
                src_ind += source_comp.dimension
            else:
                raise AttributeError("Cannont find component "+str(comp)+" in source_rv.components.")
        return ret

    def __str__(self):
        return "<pybayes.pdfs.RV '{0}' dim={1} {2}>".format(self.name, self.dimension, self.components)


class CPdf(object):
    r"""Base class for all Conditional (in general) Probability Density Functions.

    When you evaluate a CPdf the result generally also depends on a condition
    (vector) named `cond` in PyBayes. For a CPdf that is a :class:`Pdf` this is
    not the case, the result is unconditional.

    Every CPdf takes (apart from others) 2 optional arguments to constructor:
    **rv** (:class:`RV`) and **cond_rv** (:class:`RV`). When specified, they
    denote that the CPdf is associated with particular random variable (respectively
    its condition is associated with particular random variable); when unspecified,
    *anonymous* random variable is assumed (exceptions exist, see :class:`ProdPdf`).
    It is an error to pass RV whose dimension is not same as CPdf's dimension
    (or cond dimension respectively).

    :var RV rv: associated random variable (always set in constructor, contains
       at least one RVComp)
    :var RV cond_rv: associated condition random variable (set in constructor to
       potentially empty RV)

    *While you can assign different rv and cond_rv to a CPdf, you should be
    cautious because sanity checks are only performed in constructor.*

    While entire idea of random variable associations may not be needed in simple
    cases, it allows you to express more complicated situations, assume the state
    variable is composed of 2 components :math:`x_t = [a_t, b_t]` and following
    probability density function has to be modelled:

    .. math::

       p(x_t|x_{t-1}) &:= p_1(a_t|a_{t-1}, b_t) p_2(b_t|b_{t-1}) \\
       p_1(a_t|a_{t-1}, b_t) &:= \mathcal{N}(a_{t-1}, b_t) \\
       p_2(b_t|b_{t-1}) &:= \mathcal{N}(b_{t-1}, 0.0001)

    This is done in PyBayes with associated RVs:

    >>> a_t, b_t = RVComp(1, 'a_t'), RVComp(1, 'b_t')  # create RV components
    >>> a_tp, b_tp = RVComp(1, 'a_{t-1}'), RVComp(1, 'b_{t-1}')  # t-1 case

    >>> p1 = LinGaussPdf(1., 0., 1., 0., RV(a_t), RV(a_tp, b_t))
    >>> cov, A, b = np.array([[0.0001]]), np.array([[1.]]), np.array([0.])  # params for p2
    >>> p2 = MLinGaussPdf(cov, A, b, RV(b_t), RV(b_tp))

    >>> p = ProdCPdf((p1, p2), RV(a_t, b_t), RV(a_tp, b_tp))

    >>> p.sample(np.array([1., 2.]))
    >>> p.eval_log()
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

        :rtype: :class:`numpy.ndarray`"""
        raise NotImplementedError("Derived classes must implement this function")

    def variance(self, cond = None):
        """Return (conditional) variance (diagonal elements of covariance).

        :rtype: :class:`numpy.ndarray`"""
        raise NotImplementedError("Derived classes must implement this function")

    def eval_log(self, x, cond = None):
        """Return logarithm of (conditional) likelihood function in point x.

        :param x: point which to evaluate the function in
        :type x: :class:`numpy.ndarray`
        :rtype: double"""
        raise NotImplementedError("Derived classes must implement this function")

    def sample(self, cond = None):
        """Return one random (conditional) sample from this distribution

        :rtype: :class:`numpy.ndarray`"""
        raise NotImplementedError("Derived classes must implement this function")

    def samples(self, n, cond = None):
        """Return n samples in an array. A convenience function that just calls
        :meth:`shape` multiple times.

        :param int n: number of samples to return
        :rtype: 2D :class:`numpy.ndarray` of shape (*n*, m) where m is pdf
           dimension"""
        ret = empty((n, self.shape()))
        for i in range(n):
            ret[i] = self.sample(cond)
        return ret

    def _check_cond(self, cond):
        """Return True if cond has correct type and shape, raise Error otherwise.

        :raises TypeError: cond is not of correct type
        :raises ValueError: cond doesn't have appropriate shape
        :rtype: bool"""
        if cond is None:  # cython-specific
            raise TypeError("cond must be numpy.ndarray")
        if cond.ndim != 1:
            raise ValueError("cond must be 1D numpy array (a vector)")
        if cond.shape[0] != self.cond_shape():
            raise ValueError("cond must be of shape ({0},) array of shape ({1},) given".format(self.cond_shape(), cond.shape[0]))
        return True

    def _check_x(self, x):
        """Return True if x has correct type and shape (determined by shape()),
        raise Error otherwise.

        :raises TypeError: cond is not of correct type
        :raises ValueError: cond doesn't have appropriate shape
        :rtype: bool"""
        if x is None:  # cython-specific
            raise TypeError("x must be numpy.ndarray")
        if x.ndim != 1:
            raise ValueError("x must be 1D numpy array (a vector)")
        if x.shape[0] != self.shape():
            raise ValueError("x must be of shape ({0},) array of shape ({1},) given".format(self.shape(), x.shape[0]))
        return True

    def _set_rvs(self, rv, cond_rv):
        """Internal heper to check and set rv and cond_rv.

        :raises TypeError: rv or cond_rv doesnt have right type
        :raises ValueError: dimensions do not match"""
        if rv is None:
            self.rv = RV(RVComp(self.shape()))  # create RV with one anonymous component
        else:
            if not isinstance(rv, RV):
                raise TypeError("rv (is specified) must be (a subclass of) RV")
            if rv.dimension != self.shape():
                raise ValueError("rv has wrong dimension " + str(rv.dimension) + ", " + str(self.shape()) + " expected")
            self.rv = rv

        if cond_rv is None:
            if self.cond_shape() is 0:
                self.cond_rv = RV()  # create empty RV to denote empty condition
            else:
                self.cond_rv = RV(RVComp(self.cond_shape()))  # create RV with one anonymous component
        else:
            if not isinstance(cond_rv, RV):
                raise TypeError("cond_rv (is specified) must be (a subclass of) RV")
            if cond_rv.dimension is not self.cond_shape():
                raise ValueError("cond_rv has wrong dimension " + str(cond_rv.dimension) + ", " + str(self.cond_shape()) + " expected")
            self.cond_rv = cond_rv
        return True


class Pdf(CPdf):
    """Base class for all unconditional (static) multivariate Probability Density
    Functions. Subclass of :class:`CPdf`.

    As in CPdf, constructor of every Pdf takes optional **rv** (:class:`RV`)
    keyword argument (and no *cond_rv* argument as it would make no sense). For
    discussion about associated random variables see :class:`CPdf`.
    """

    def cond_shape(self):
        """Return zero as Pdfs have no condition."""
        return 0


class UniPdf(Pdf):
    r"""Simple uniform multivariate probability density function.

    .. math:: f(x) = \Theta(x - a) \Theta(b - x) \prod_{i=1}^n \frac{1}{b_i-a_i}

    :var a: left border
    :type a: :class:`numpy.ndarray`
    :var b: right border
    :type b: :class:`numpy.ndarray`

    You may modify these attributes as long as you don't change their shape and
    assumption **a** < **b** still holds.
    """

    def __init__(self, a, b, rv = None):
        """Initialise uniform distribution.

        :param a: left border
        :type a: :class:`numpy.ndarray`
        :param b: right border
        :type b: :class:`numpy.ndarray`

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
        self._set_rvs(rv, None)

    def shape(self):
        return self.a.shape[0]

    def mean(self, cond = None):
        return (self.a+self.b)/2.  # element-wise division

    def variance(self, cond = None):
        return ((self.b-self.a)**2)/12.  # element-wise power and division

    def eval_log(self, x, cond = None):
        self._check_x(x)
        if np_any(x <= self.a) or np_any(x >= self.b):
            return float('-inf')
        return -log(prod(self.b-self.a))

    def sample(self, cond = None):
        return uniform(-0.5, 0.5, self.shape()) * (self.b-self.a) + self.mean()


class GaussPdf(Pdf):
    r"""Unconditional Gaussian (normal) probability density function.

    .. math:: f(x) \propto \exp \left( - \left( x-\mu \right)' R^{-1} \left( x-\mu \right) \right)

    :var mu: mean value
    :type mu: 1-D :class:`numpy.ndarray`
    :var R: covariance matrix
    :type R: 2-D :class:`numpy.ndarray`

    You can modify object parameters only if you are absolutely sure that you
    pass allowable values, because parameters are only checked once in constructor.
    """

    def __init__(self, mean, covariance, rv = None):
        """Initialise Gaussian pdf.

        :param mean: mean value
        :type mean: 1-D :class:`numpy.ndarray`
        :param covariance: covariance matrix
        :type covariance: 2-D :class:`numpy.ndarray`

        To create standard normal distribution:

        >>> norm = GaussPdf(np.array([0.]), np.array([[1.]]))  # note the shape of covariance
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
        self._set_rvs(rv, None)

    def shape(self):
        return self.mu.shape[0]

    def mean(self, cond = None):
        return self.mu

    def variance(self, cond = None):
        return diag(self.R)

    def eval_log(self, x, cond = None):
        self._check_x(x)

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


class EmpPdf(Pdf):
    r"""Weighted empirical probability density function.

    .. math::

       p(x) &= \sum_{i=1}^n \omega_i \delta(x - x^{(i)}) \\
       \text{where} \quad x^{(i)} &\text{ is value of the i}^{th} \text{ particle} \\
       \omega_i \geq 0 &\text{ is weight of the i}^{th} \text{ particle} \quad \sum \omega_i = 1

    :var numpy.ndarray particles: 2D array of particles; shape: (n, m) where n
       is the number of particles, m dimension of this pdf
    :var numpy.ndarray weights: 1D array of particle weights

    You may alter particles and weights, but you must ensure that their shapes
    match and that weight constraints still hold. You can use
    :meth:`normalise_weights` to do some work for you.
    """

    def __init__(self, init_particles, rv = None):
        r"""Initialise empirical pdf.

        :param init_particles: 2D array of initial particles; shape (*n*, *m*)
           determines that *n* *m*-dimensioned particles will be used
        :type init_particles: :class:`numpy.ndarray`
        """
        self.particles = init_particles
        # set n weights to 1/n
        self.weights = ones(self.particles.shape[0]) / self.particles.shape[0]

        self._set_rvs(rv, None)

    def shape(self):
        return self.particles.shape[0]

    def mean(self, cond = None):
        ret = zeros(self.particles.shape[1])
        for i in range(self.particles.shape[0]):
            ret += self.weights[i] * self.particles[i]
        return ret

    def variance(self, cond = None):
        ret = zeros(self.particles.shape[1])
        for i in range(self.particles.shape[0]):
            ret += self.weights[i] * (self.particles[i])**2
        return ret - self.mean()**2

    def eval_log(self, x, cond = None):
        raise NotImplementedError("eval_log doesn't make sense for discrete distribution")

    def sample(self, cond = None):
        raise NotImplementedError("Sample for empirical pdf not (yet?) implemented")

    def normalise_weights(self):
        r"""Multiply weights by appropriate constant so that
        :math:`\sum \omega_i = 1`

        :raise AttributeError: when :math:`\exists i: \omega_i < 0` or
           :math:`\forall i: \omega_i = 0`
        """
        if np_any(self.weights < 0.):
            raise AttributeError("Weights must not be negative")
        wsum = sum(self.weights)
        if wsum == 0:
            raise AttributeError("Sum of weights == 0: weights cannot be normalised")
        self.weights *= 1./wsum


class ProdPdf(Pdf):
    r"""Unconditional product of multiple unconditional pdfs.

    You can for example create a pdf that has uniform distribution with regards
    to x-axis and normal distribution along y-axis. The caller (you) must ensure
    that individial random variables are independent, otherwise their product may
    have no mathematical sense.

    .. math:: f(x_1 x_2 x_3) = f_1(x_1) f_2(x_2) f_3(x_3)
    """

    def __init__(self, factors, rv = None):
        r"""Initialise product of unconditional pdfs.

        :param factors: sequence of sub-distributions
        :type factors: sequence of :class:`Pdf`

        As an exception from the general rule, ProdPdf does not create anonymous
        associated random variable if you do not supply it in constructor - it
        rather reuses components of underlying factor pdfs. (You can of course
        override this behaviour by bassing custom **rv**.)

        Usual way of creating ProdPdf could be:

        >>> prod = ProdPdf((UniPdf(...), GaussPdf(...)))  # note the double (( and ))
        """
        if rv is None:
            rv_comps = []  # prepare to construnct associated rv
        else:
            rv_comps = None

        if len(factors) is 0:
            raise ValueError("at least one factor must be passed")
        self.factors = array(factors, dtype=Pdf)
        self.shapes = zeros(self.factors.shape[0], dtype=int)  # array of factor shapes
        for i in range(self.factors.shape[0]):
            if not isinstance(self.factors[i], Pdf):
                raise TypeError("all records in factors must be (subclasses of) Pdf")
            self.shapes[i] = self.factors[i].shape()
            if rv_comps is not None:
                rv_comps.extend(self.factors[i].rv.components)  # add components of child rvs

        # pre-calclate shape
        self._shape = sum(self.shapes)
        # associate with a rv (needs to be after _shape calculation)
        if rv_comps is None:
            self._set_rvs(rv, None)
        else:
            self._set_rvs(RV(*rv_comps), None)

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
        self._check_x(x)

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
    r"""Conditional Gaussian pdf whose mean is a linear function of condition.

    .. math::

       f(x|c) \propto \exp \left( - \left( x-\mu \right)' R^{-1} \left( x-\mu \right) \right)
       \quad \quad \text{where} ~ \mu := A c + b
    """

    def __init__(self, covariance, A, b, rv = None, cond_rv = None):
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
        self._set_rvs(rv, cond_rv)

    def shape(self):
        return self.b.shape[0]

    def cond_shape(self):
        return self.A.shape[1]

    def mean(self, cond = None):
        self._check_cond(cond)
        return dot(self.A, cond) + self.b

    def variance(self, cond = None):
        # cond is unused here, so no need to check it
        return self.gauss.variance()

    def eval_log(self, x, cond = None):
        # x is checked in self.gauss
        # cond is checked in mean()
        self.gauss.mu = self.mean(cond)
        return self.gauss.eval_log(x)

    def sample(self, cond = None):
        # cond is checked in mean()
        self.gauss.mu = self.mean(cond)
        return self.gauss.sample()


class LinGaussCPdf(CPdf):
    r"""Conditional one-dimensional Gaussian pdf whose mean and covariance are
    linear functions of condition.

    .. math::

       f(x|c_1 c_2) \propto \exp \left( - \frac{\left( x-\mu \right)^2}{2\sigma^2} \right)
       \quad \quad \text{where} \quad \mu := a c_1 + b \quad \text{and}
       \quad \sigma^2 := c c_2 + d
    """

    def __init__(self, a, b, c, d, rv = None, cond_rv = None):
        """Initialise Linear Gaussian conditional pdf.

        :param double a, b: mean = a*cond_1 + b
        :param double c, d: covariance = c*cond_2 + d
        """
        if not isinstance(a, float):
            raise TypeError("all parameters must be floats")
        self.a = a
        if not isinstance(b, float):
            raise TypeError("all parameters must be floats")
        self.b = b
        if not isinstance(c, float):
            raise TypeError("all parameters must be floats")
        self.c = c
        if not isinstance(d, float):
            raise TypeError("all parameters must be floats")
        self.d = d
        self.gauss = GaussPdf(zeros(1), array([[1.]]))
        self._set_rvs(rv, cond_rv)

    def shape(self):
        return 1

    def cond_shape(self):
        return 2

    def mean(self, cond = None):
        self._check_cond(cond)
        self.gauss.mu[0] = self.a*cond[0] + self.b  # gauss.mu is used just as a holder
        return self.gauss.mu

    def variance(self, cond = None):
        self._check_cond(cond)
        return array([self.c*cond[1] + self.d])

    def eval_log(self, x, cond = None):
        # x is checked in self.gauss
        self._check_cond(cond)
        self.gauss.mu[0] = self.a*cond[0] + self.b
        self.gauss.R[0,0] = self.c*cond[1] + self.d
        return self.gauss.eval_log(x)

    def sample(self, cond = None):
        self._check_cond(cond)
        self.gauss.mu[0] = self.a*cond[0] + self.b
        self.gauss.R[0,0] = self.c*cond[1] + self.d
        return self.gauss.sample()


class ProdCPdf(CPdf):
    r"""Pdf that is formed as a chain rule of multiple conditional pdfs. In a
    simple textbook case denoted below it isn't needed to specify random variables
    at all. In this case when no random variable associations are passed,
    ProdCPdf ignores rv associations of its factors and everything is determined
    from their order. (:math:`x_i` are arbitrary vectors)

    .. math::

        f(x_1 x_2 x_3 | c) &= f_1(x_1 | x_2 x_3 c) f_2(x_2 | x_3 c) f_3(x_3 | c) \\
        \text{or} \quad f(x_1 x_2 x_3) &= f_1(x_1 | x_2 x_3) f_2(x_2 | x_3) f_3(x_3)

    >>> f = ProdCPdf((f1, f2, f3))

    For less simple situations, specifiying random value associations is needed
    to estabilish data chain:

    .. math:: p(x_1 x_2 | y_1 y_2) = p_1(x_1 | x_2) p_2(x_2 | y_2 y_1)

    >>> # prepare random variable components:
    >>> x_1, x_2 = RVComp(1), RVComp(1, "name is optional")
    >>> y_1, y_2 = RVComp(1), RVComp(1, "but recommended")

    >>> p_1 = SomePdf(..., rv=RV(x_1), cond_rv=RV(x_2))
    >>> p_2 = SomePdf(..., rv=RV(x_2), cond_rv=RV(y_2, y_1))
    >>> p = ProdCPdf((p_2, p_1), rv=RV(x_1, x_2), cond_rv=RV(y_1, y_2))  # order of
    >>> # pdfs is insignificant - order of rv components determines data flow
    """

    def __init__(self, factors, rv = None, cond_rv = None):
        """Construct chain rule of multiple cpdfs.

        :param factors: sequence of densities that will form the product
        :type factors: sequence of :class:`CPdf`

        Usual way of creating ProdCPdf could be:

        >>> prod = ProdCPdf((MLinGaussCPdf(..), UniPdf(..)), RV(..), RV(..))
        """
        if len(factors) is 0:
            raise ValueError("at least one factor must be passed")

        self.in_indeces = []  # data link representations
        self.out_indeces = []

        if rv is None and cond_rv is None:
            self._init_anonymous(factors)
        elif rv is not None and cond_rv is not None:
            self._init_with_rvs(list(factors), rv, cond_rv)  # needs factors as list
        else:
            raise AttributeError("Please pass both rv and cond_rv or none of them, other combinations not (yet) supported")

        self._set_rvs(rv, cond_rv)

    def _init_anonymous(self, factors):
        self.factors = array(factors, dtype=CPdf)

        # overall cond shape equals last factor cond shape:
        self._cond_shape = factors[-1].cond_shape()
        self._shape = factors[0].shape() + factors[0].cond_shape() - self._cond_shape

        start_ind = 0  # current start index in cummulate rv and cond_rv data array
        for i in range(self.factors.shape[0]):
            factor = self.factors[i]
            if not isinstance(factor, CPdf):
                raise TypeError("all records in factors must be (subclasses of) CPdf")

            shape = factor.shape()
            cond_shape = factor.cond_shape()
            # expected (normal + cond) shape:
            exp_shape = self._shape + self._cond_shape - start_ind
            if shape + cond_shape != exp_shape:
                raise ValueError("Expected that pdf {0} will have shape (={1}) + ".
                    format(factor, shape) + "cond_shape (={0}) == {1}".
                    format(cond_shape, exp_shape))

            self.in_indeces.append(arange(start_ind + shape, start_ind + shape + cond_shape))
            self.out_indeces.append(arange(start_ind, start_ind + shape))

            start_ind += shape

        if start_ind != self._shape:
            raise ValueError("Shapes do not match")

    def _init_with_rvs(self, factors, rv, cond_rv):
        """Initialise ProdCPdf using rv components for data chain construction.

        :param factors: factor pdfs that will form the product
        :type factors: :class:`list` of :class:`CPdf` items
        """
        # gradually filled set of components that would be available in e.g.
        # sample() computation:
        avail_rvcomps = set(cond_rv.components)

        self.factors = empty(len(factors), dtype=CPdf)  # initialise factor array

        i = self.factors.shape[0] - 1  # factors are filled from right to left
        # iterate until all input pdfs are processed
        while len(factors) > 0:
            # find next pdf that can be added to data chain (all its cond
            # components can be already computed)
            for j in range(len(factors)):
                factor = factors[j]
                if not isinstance(factor, CPdf):
                    raise TypeError("all records in factors must be (subclasses of) CPdf")
                if factor.cond_rv.contained_in(avail_rvcomps):
                    # one such pdf found
                    #DEBUG: print "Appropriate pdf found:", factor, "with rv:", factor.rv, "and cond_rv:", factor.cond_rv
                    if not rv.contains_all(factor.rv.components):
                        raise AttributeError(("Some of {0}'s associated rv components "
                            + "({1}) aren't present in rv ({2})").format(factor, factor.rv, rv))
                    avail_rvcomps.update(factor.rv.components)
                    self.factors[i] = factor
                    i += -1
                    del factors[j]
                    break;
            else:
                # we are stuck somewhere in data chain
                print "Appropriate pdf not found. avail_rvcomps:", avail_rvcomps, "candidates:"
                for factor in factors:
                    print "  ", factor, "with cond_rv:", factor.cond_rv
                raise AttributeError("Cannont construct data chain. This means "
                    + "that it is impossible to arrange factor pdfs into a DAG "
                    + "that starts with ProdCPdf's cond_rv components. Please "
                    + "check cond_rv and factor rvs and cond_rvs.")
        if not rv.contained_in(avail_rvcomps):
            print "These components can be computed:", avail_rvcomps
            print "... but we have to fill following rv:", rv
            raise AttributeError("Data chain built, some components cannot be "
                + "computed with it.")

        cummulate_rv = RV(rv, cond_rv)
        for i in range(self.factors.shape[0]):
            factor = self.factors[i]
            self.in_indeces.append(factor.cond_rv.indexed_in(cummulate_rv))
            self.out_indeces.append(factor.rv.indexed_in(cummulate_rv))

        self._shape = rv.dimension
        self._cond_shape = cond_rv.dimension

    def shape(self):
        return self._shape

    def cond_shape(self):
        return self._cond_shape

    def mean(self, cond = None):
        raise NotImplementedError("Not yet implemented")

    def variance(self, cond = None):
        raise NotImplementedError("Not yet implemented")

    def eval_log(self, x, cond = None):
        self._check_x(x)
        self._check_cond(cond)

        # combination of evaluation point and condition:
        data = empty(self._shape + self._cond_shape)
        data[0:self._shape] = x
        data[self._shape:] = cond
        ret = 0.

        for i in range(self.factors.shape[0]):
            ret += self.factors[i].eval_log(data[self.out_indeces[i]], data[self.in_indeces[i]])
        return ret

    def sample(self, cond = None):
        self._check_cond(cond)

        # combination of sampled variables and condition:
        data = empty(self._shape + self._cond_shape)
        data[self._shape:] = cond  # rest is undefined

        # process pdfs from right to left (they are arranged so that data flow
        # is well defined in this case):
        for i in range(self.factors.shape[0] -1, -1, -1):
            data[self.out_indeces[i]] = self.factors[i].sample(data[self.in_indeces[i]])

        return data[:self._shape]  # return right portion of data
