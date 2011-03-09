# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
This module contains Bayesian filters.

All classes from this module are currently imported to top-level pybayes module,
so instead of ``from pybayes.filters import KalmanFilter`` you can type ``from
pybayes import KalmanFilter``.
"""


from numpywrap import *

from pybayes.pdfs import GaussPdf


class Filter(object):
    """Abstract prototype of a bayesian filter"""

    def bayes(self, yt, ut = None):
        """Approximate or exact bayes rule (one iteration)

        :param yt: observation at time t
        :type yt: :class:`numpy.ndarray`
        :param ut: intervence at time t (appliciable only to some filters)
        :type ut: :class:`numpy.ndarray`
        """
        raise NotImplementedError("Derived classes must implement this method")


class KalmanFilter(Filter):
    """Kalman filter"""

    def __init__(self, A, B, C, D, Q, R, state_pdf):
        """TODO: documentation"""

        # check type of pdf
        if not isinstance(state_pdf, GaussPdf):
            raise TypeError("state_pdf must be (a subclass of) GaussPdf")

        # check type of input arrays
        matrices = {"A":A, "B":B, "C":C, "D":D, "Q":Q, "R":R}
        for name in matrices:
            matrix = matrices[name]
            if type(matrix) != ndarray:  # TODO: insinstance(), but has different semantics
                raise TypeError(name + " must be (exactly) numpy.ndarray, " +
                                str(type(matrix)) + " given")
            if matrix.ndim != 2:
                raise ValueError(name + " must have 2 dimensions (thus forming a matrix), " +
                                 str(matrix.ndim) + " given")

        # remember vector shapes
        self.n = state_pdf.shape()  # dimension of state vector
        self.k = B.shape[1]  # dimension of control vector
        self.j = C.shape[0]  # dimension of observation vector

        # dict of required matrice shapes (sizes)
        shapes = {
            "A":(self.n, self.n),
            "B":(self.n, self.k),
            "C":(self.j, self.n),
            "D":(self.j, self.k),
            "Q":(self.n, self.n),
            "R":(self.j, self.j)
        }
        # check input matrix sizes
        for name in matrices:
            matrix = matrices[name]
            if matrix.shape != shapes[name]:
                raise ValueError("Given shapes of state_pdf, B and C, matrix " + name +
                                 " must have shape " + str(shapes[name]) + ", " +
                                 str(matrix.shape) + " given")

        self.A, self.B, self.C, self.D, self.Q, self.R = A, B, C, D, Q, R

        self.P = state_pdf
        self.S = GaussPdf(array([0.]), array([[1.]]))  # observation probability density function

    def bayes(self, yt, ut = None):
        if not isinstance(yt, ndarray) or not isinstance(ut, ndarray):
            raise TypeError("Both yt and ut must be numpy.ndarray. " +
                            str(type(yt)) + " and " + str(type(ut)) + " given")
        if yt.ndim != 1 or yt.shape[0] != self.j:
            raise ValueError("yt must have shape " + str((self.j,)) + ". (" +
                            str(yt.shape[0]) + ",) given")  # TODO
        if ut.ndim != 1 or ut.shape[0] != self.k:
            raise ValueError("yt must have shape " + str((self.k,)) + ". (" +
                            str(ut.shape[0]) + ",) given")  # TODO

        # predict
        self.P.mu = dot(self.A, self.P.mu) + dot(self.B, ut)  # a priori estimate
        self.P.R  = dot(dot(self.A, self.P.R), self.A.T) + self.Q  # a priori variance

        # data update
        self.S.mu = dot(self.C, self.P.mu) + dot(self.D, ut)
        self.S.R = dot(dot(self.C, self.P.R), self.C.T) + self.R

        # kalman gain
        K = dot(dot(self.P.R, self.C.T), inv(self.S.R))

        self.P.mu += dot(K, (yt - self.S.mu))  # a posteriori estimate
        self.P.R -= dot(dot(K, self.C), self.P.R)  # a posteriori variance

        return self.P


class ParticleFilter(Filter):
    r"""A filter whose aposteriori density takes the form

    .. math:: p(x_t|y_{1:t}) = \sum_{i=1}^n \omega_i \delta ( x_t - x_t^{(i)} )
    """

    def __init__(self, n, init_pdf, p_xt_xtp, p_xt_yt):
        r"""Initialise particle filter.

        :param int n: number of particles
        :param init_pdf: probability density which initial particles are sampled from
        :type init_pdf: :class:`~pybayes.pdfs.Pdf`
        :param p_xt_xtp: :math:`p(x_t|x_{t-1})` pdf of state in *t* given state in *t-1*
        :type p_xt_xtp: :class:`~pybayes.pdfs.CPdf`
        :param p_xt_yt: :math:`p(x_t|y_t)` pdf of state in *t* given observation in *t*
        :type p_xt_yt: :class:`~pybayes.pdfs.CPdf`
        """
        dim = init_pdf.shape()  # dimension of state
        if p_xt_xtp.shape() != dim or p_xt_xtp.cond_shape() != dim:
            raise ValueError("Expected shape() and cond_shape() of p_xt_xtp() will "
                + "be {0}; ({1}, {2}) given.".format(dim, p_xt_xtp.shape(),
                p_xt_xtp.cond_shape()))
        if p_xt_yt.shape() != dim:
            raise ValueError("Expected shape() of p_xt_yt() will be {0}; {1} given."
                .format(dim, p_xt_yt.shape()))

        # generate initial particles:
        self.particles = empty((n, dim))
        for i in range(n):
            self.particles[i] = init_pdf.sample()
