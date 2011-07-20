# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
This module contains Bayesian filters.

All classes from this module are currently imported to top-level pybayes module,
so instead of ``from pybayes.filters import KalmanFilter`` you can type ``from
pybayes import KalmanFilter``.
"""

from copy import deepcopy
from math import exp

import wrappers._linalg as linalg
import wrappers._numpy as np
from pybayes.pdfs import CPdf, Pdf, GaussPdf, EmpPdf, MarginalizedEmpPdf


class Filter(object):
    """Abstract prototype of a bayesian filter."""

    def bayes(self, yt, cond = None):
        """Perform approximate or exact bayes rule.

        :param yt: observation at time t
        :type yt: 1D :class:`numpy.ndarray`
        :param cond: condition at time t. Exact meaning is defined by each filter
        :type cond: 1D :class:`numpy.ndarray`
        :return: always returns True (see :meth:`posterior` to get posterior density)
        """
        raise NotImplementedError("Derived classes must implement this method")

    def posterior(self):
        """Return posterior probability density funcion (:class:`~pybayes.pdfs.Pdf`).

        :return: posterior density
        :rtype: :class:`~pybayes.pdfs.Pdf`

        *Filter implementations may decide to return a reference to their work pdf - it is not safe
        to modify it in any way, doing so may leave the filter in undefined state.*
        """
        raise NotImplementedError("Derived classes must implement this method")

    def evidence_log(self, yt):
        """Return the logarithm of *evidence* function (also known as *marginal likehood*) evaluated
        in point yt.

        :param yt: point which to evaluate the evidence in
        :type yt: :class:`numpy.ndarray`
        :rtype: double

        This is typically computed after :meth:`bayes` with the same observation:

        >>> filter.bayes(yt)
        >>> log_likehood = filter.evidence_log(yt)
        """
        raise NotImplementedError("Derived classes should implement this method, if feasible")


class KalmanFilter(Filter):
    r"""Implementation of standard Kalman filter. **cond** in :meth:`bayes` is interpreted as
    control (intervention) input :math:`u_t` to the system.

    Kalman filter forms *optimal Bayesian solution* for the following system:

    .. math::

        x_t &= A_t x_{t-1} + B_t u_t + v_{t-1} \quad \quad
            A_t \in \mathbb{R}^{n,n} \;\;
            B_t \in \mathbb{R}^{n,k} \;\;
            \;\; n \in \mathbb{N}
            \;\; k \in \mathbb{N}_0 \text{ (may be zero)}
            \\
        y_t &= C_t x_t + D_t u_t + w_t \quad \quad \quad \;\;
            C_t \in \mathbb{R}^{j,n} \;\;
            D_t \in \mathbb{R}^{j,k} \;\;
            j \in \mathbb{N} \;\; j \leq n

    where :math:`x_t \in \mathbb{R}^n` is hidden state vector, :math:`y_t \in \mathbb{R}^j` is
    observation vector and :math:`u_t \in \mathbb{R}^k` is control vector. :math:`v_t` is normally
    distributed zero-mean process noise with covariance matrix :math:`Q_t`, :math:`w_t` is normally
    distributed zero-mean observation noise with covariance matrix :math:`R_t`. Additionally, intial
    pdf (**state_pdf**) has to be Gaussian.
    """

    def __init__(self, A, B = None, C = None, D = None, Q = None, R = None, state_pdf = None):
        r"""Initialise Kalman filter.

        :param A: process model matrix :math:`A_t` from :class:`class description <KalmanFilter>`
        :type A: 2D :class:`numpy.ndarray`
        :param B: process control model matrix :math:`B_t` from :class:`class description
           <KalmanFilter>`; may be None or unspecified for control-less systems
        :type B: 2D :class:`numpy.ndarray`
        :param C: observation model matrix :math:`C_t` from :class:`class description
           <KalmanFilter>`; must be full-ranked
        :type C: 2D :class:`numpy.ndarray`
        :param D: observation control model matrix :math:`D_t` from :class:`class description
           <KalmanFilter>`; may be None or unspecified for control-less systems
        :type D: 2D :class:`numpy.ndarray`
        :param Q: process noise covariance matrix :math:`Q_t` from :class:`class description
           <KalmanFilter>`; must be positive definite
        :type Q: 2D :class:`numpy.ndarray`
        :param R: observation noise covariance matrix :math:`R_t` from :class:`class description
           <KalmanFilter>`; must be positive definite
        :type R: 2D :class:`numpy.ndarray`
        :param state_pdf: initial state pdf; this object is referenced and used throughout whole
           life of KalmanFilter, so it is not safe to reuse state pdf for other purposes
        :type state_pdf: :class:`~pybayes.pdfs.GaussPdf`

        All matrices can be time-varying - you can modify or replace all above stated matrices
        providing that you don't change their shape and all constraints still hold. On the other
        hand, you **should not modify state_pdf** unless you really know what you are doing.

        >>> # initialise control-less Kalman filter:
        >>> kf = pb.KalmanFilter(A=np.array([[1.]]),
                                 C=np.array([[1.]]),
                                 Q=np.array([[0.7]]), R=np.array([[0.3]]),
                                 state_pdf=pb.GaussPdf(...))
        """

        # check type of pdf
        if not isinstance(state_pdf, GaussPdf):
            raise TypeError("state_pdf must be (a subclass of) GaussPdf")

        # check type of input arrays
        matrices = {"A":A, "B":B, "C":C, "D":D, "Q":Q, "R":R}
        for name in matrices:
            matrix = matrices[name]
            if name == 'B' and matrix is None:  # we allow B to be unspecified
                matrices['B'] = matrix = B = np.empty((A.shape[0], 0))
            if name == 'D' and matrix is None:  # we allow D to be unspecified
                matrices['D'] = matrix = D = np.empty((C.shape[0], 0))

            if not isinstance(matrix, np.ndarray):
                raise TypeError("{0} must be numpy.ndarray, {1} given".format(name, type(matrix)))
            if matrix.ndim != 2:
                raise ValueError("{0} must have 2 dimensions (forming a matrix) {1} given".format(
                                 name, matrix.ndim))

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
        self.S = GaussPdf(np.array([0.]), np.array([[1.]]))  # observation probability density function

    def __copy__(self):
        # type(self) is used because this method may be called for a derived class
        ret = type(self).__new__(type(self))
        ret.A = self.A
        ret.B = self.B
        ret.C = self.C
        ret.D = self.D
        ret.Q = self.Q
        ret.R = self.R
        ret.n = self.n
        ret.k = self.k
        ret.j = self.j
        ret.P = self.P
        ret.S = self.S
        return ret

    def __deepcopy__(self, memo):
        # type(self) is used because this method may be called for a derived class
        ret = type(self).__new__(type(self))
        ret.A = deepcopy(self.A, memo)  # numpy arrays:
        ret.B = deepcopy(self.B, memo)
        ret.C = deepcopy(self.C, memo)
        ret.D = deepcopy(self.D, memo)
        ret.Q = deepcopy(self.Q, memo)
        ret.R = deepcopy(self.R, memo)
        ret.n = self.n  # no need to copy integers
        ret.k = self.k
        ret.j = self.j
        ret.P = deepcopy(self.P, memo)  # GaussPdfs:
        ret.S = deepcopy(self.S, memo)
        return ret

    def bayes(self, yt, cond = np.empty(0)):
        r"""Perform exact bayes rule.

        :param yt: observation at time t
        :type yt: 1D :class:`numpy.ndarray`
        :param cond: control (intervention) vector at time t. May be unspecified if the filter is
           control-less.
        :type cond: 1D :class:`numpy.ndarray`
        :return: always returns True (see :meth:`~Filter.posterior` to get posterior density)
        """
        if not isinstance(yt, np.ndarray):
            raise TypeError("yt must be and instance of numpy.ndarray ({0} given)".format(type(yt)))
        if yt.ndim != 1 or yt.shape[0] != self.j:
            raise ValueError("yt must have shape {0}. ({1} given)".format((self.j,), (yt.shape[0],)))
        if not isinstance(cond, np.ndarray):
            raise TypeError("cond must be and instance of numpy.ndarray ({0} given)".format(type(cond)))
        if cond.ndim != 1 or cond.shape[0] != self.k:
            raise ValueError("cond must have shape {0}. ({1} given)".format((self.k,), (cond.shape[0],)))

        # predict
        self.P.mu = np.dot(self.A, self.P.mu) + np.dot(self.B, cond)  # prior state mean estimate
        self.P.R  = np.dot(np.dot(self.A, self.P.R), self.A.T) + self.Q  # prior state covariance estimate

        # data update
        self.S.mu = np.dot(self.C, self.P.mu) + np.dot(self.D, cond)  # prior observation mean estimate
        self.S.R = np.dot(np.dot(self.C, self.P.R), self.C.T) + self.R  # prior observation covariance estimate

        # kalman gain
        K = np.dot(np.dot(self.P.R, self.C.T), linalg.inv(self.S.R))

        # update according to observation
        self.P.mu += np.dot(K, (yt - self.S.mu))  # posterior state mean estimate
        self.P.R -= np.dot(np.dot(K, self.C), self.P.R)  # posterior state covariance estimate
        return True

    def posterior(self):
        return self.P

    def evidence_log(self, yt):
        return self.S.eval_log(yt)


class ParticleFilter(Filter):
    r"""Standard particle filter implementation with resampling.

    Specifying proposal density is currently unsupported, but planned; speak up if you want it!
    Posterior pdf is represented using :class:`~pybayes.pdfs.EmpPdf` and takes following form:

    .. math:: p(x_t|y_{1:t}) = \sum_{i=1}^n \omega_i \delta ( x_t - x_t^{(i)} )
    """

    def __init__(self, n, init_pdf, p_xt_xtp, p_yt_xt):
        r"""Initialise particle filter.

        :param int n: number of particles
        :param init_pdf: probability density which initial particles are sampled from
        :type init_pdf: :class:`~pybayes.pdfs.Pdf`
        :param p_xt_xtp: :math:`p(x_t|x_{t-1})` cpdf of state in *t* given state in *t-1*
        :type p_xt_xtp: :class:`~pybayes.pdfs.CPdf`
        :param p_yt_xt: :math:`p(y_t|x_t)` cpdf of observation in *t* given state in *t*
        :type p_yt_xt: :class:`~pybayes.pdfs.CPdf`
        """
        if not isinstance(n, int) or n < 1:
            raise TypeError("n must be a positive integer")
        if not isinstance(init_pdf, Pdf):
            raise TypeError("init_pdf must be an instance ot the Pdf class")
        if not isinstance(p_xt_xtp, CPdf) or not isinstance(p_yt_xt, CPdf):
            raise TypeError("both p_xt_xtp and p_yt_xt must be instances of the CPdf class")

        dim = init_pdf.shape()  # dimension of state
        if p_xt_xtp.shape() != dim or p_xt_xtp.cond_shape() != dim:
            raise ValueError("Expected shape() and cond_shape() of p_xt_xtp will "
                + "be {0}; ({1}, {2}) given.".format(dim, p_xt_xtp.shape(),
                p_xt_xtp.cond_shape()))
        self.p_xt_xtp = p_xt_xtp
        if p_yt_xt.cond_shape() != dim:
            raise ValueError("Expected cond_shape() of p_yt_xt will be {0}; {1} given."
                .format(dim, p_yt_xt.cond_shape()))
        self.p_yt_xt = p_yt_xt

        # generate initial particles:
        self.emp = EmpPdf(init_pdf.samples(n))

    def bayes(self, yt, cond = None):
        r"""Perform Bayes rule for new measurement :math:`y_t`. The algorithm is as follows:

        1. generate new particles: :math:`x_t^{(i)} = \text{sample from }
           p(x_t^{(i)}|x_{t-1}^{(i)}) \quad \forall i`
        2. recompute weights: :math:`\omega_i = p(y_t|x_t^{(i)})
           \omega_i \quad \forall i`
        3. normalise weights
        4. resample particles
        """
        for i in range(self.emp.particles.shape[0]):
            # generate new ith particle:
            self.emp.particles[i] = self.p_xt_xtp.sample(self.emp.particles[i])

            # recompute ith weight:
            self.emp.weights[i] *= exp(self.p_yt_xt.eval_log(yt, self.emp.particles[i]))

        # assure that weights are normalised
        self.emp.normalise_weights()
        # resample
        self.emp.resample()
        return True

    def posterior(self):
        return self.emp


class MarginalizedParticleFilter(Filter):
    r"""Simple marginalized particle filter implementation. Assume that tha state vector :math:`x`
    can be divided into two parts :math:`x_t = (a_t, b_t)` and that the pdf representing the process
    model can be factorised as follows:

    .. math:: p(x_t|x_{t-1}) = p(a_t|a_{t-1}, b_t) p(b_t | b_{t-1})

    and that the :math:`a_t` part (given :math:`b_t`) can be estimated with (a subbclass of) the
    :class:`KalmanFilter`. Such system may be suitable for the marginalized particle filter, whose
    posterior pdf takes the form

    .. math::

       p &= \sum_{i=1}^n \omega_i p(a_t | y_{1:t}, b_{1:t}^{(i)}) \delta(b_t - b_t^{(i)}) \\
       p(a_t | y_{1:t}, b_{1:t}^{(i)}) &\text{ is posterior pdf of i}^{th} \text{ Kalman filter} \\
       \text{where } \quad \quad \quad \quad \quad
       b_t^{(i)} &\text{ is value of the (b part of the) i}^{th} \text{ particle} \\
       \omega_i \geq 0 &\text{ is weight of the i}^{th} \text{ particle} \quad \sum \omega_i = 1

    *Note: currently :math:`b_t` is hard-coded to be process and observation noise covariance of the
    :math:`a_t` part. This will be changed soon and :math:`b_t` will be passed as condition to
    :meth:`KalmanFilter.bayes`.*
    """

    def __init__(self, n, init_pdf, p_bt_btp, kalman_args, kalman_class = KalmanFilter):
        r"""Initialise marginalized particle filter.

        :param int n: number of particles
        :param init_pdf: probability density which initial particles are sampled from. (both
           :math:`a_t` and :math:`b_t` parts)
        :type init_pdf: :class:`~pybayes.pdfs.Pdf`
        :param p_bt_btp: :math:`p(b_t|b_{t-1})` cpdf of the (b part of the) state in *t* given
           state in *t-1*
        :type p_bt_btp: :class:`~pybayes.pdfs.CPdf`
        :param dict kalman_args: arguments for the Kalman filter, passed as dictionary; *state_pdf*
           key should not be speficied as it is supplied by the marginalized particle filter
        :param class kalman_class: class of the filter used for the :math:`a_t` part of the system;
            defaults to :class:`KalmanFilter`
        """
        if not isinstance(n, int) or n < 1:
            raise TypeError("n must be a positive integer")
        if not isinstance(init_pdf, Pdf) or not isinstance(p_bt_btp, CPdf):
            raise TypeError("init_pdf must be a Pdf and p_bt_btp must be a CPdf")
        if not issubclass(kalman_class, KalmanFilter):
            raise TypeError("kalman_class must be a subclass (not an instance) of KalmanFilter")
        b_shape = p_bt_btp.shape()
        if p_bt_btp.cond_shape() != b_shape:
            raise ValueError("p_bt_btp's shape ({0}) and cond shape ({1}) must both be {2}".format(
                             p_bt_btp.shape(), p_bt_btp.cond_shape(), b_shape))
        self.p_bt_btp = p_bt_btp
        a_shape = init_pdf.shape() - b_shape

        # this will be removed when hardcoding Q,R into kalman filter will be removed
        kalman_args['Q'] = np.array([[-123.]])
        kalman_args['R'] = np.array([[-494658.]])

        # generate both initial parts of particles
        init_particles = init_pdf.samples(n)

        # create all Kalman filters first
        self.kalmans = np.empty(n, dtype=KalmanFilter) # array of references to Kalman filters
        gausses = np.empty(n, dtype=GaussPdf) # array of Kalman filter state pdfs
        for i in range(n):
            gausses[i] = GaussPdf(init_particles[i,0:a_shape], np.array([[1.]]))
            kalman_args['state_pdf'] = gausses[i]
            self.kalmans[i] = kalman_class(**kalman_args)
        # construct apost pdf. Important: reference to ith GaussPdf is shared between ith Kalman
        # filter's state_pdf and ith memp't gauss
        self.memp = MarginalizedEmpPdf(gausses, init_particles[:,a_shape:])

    def __str__(self):
        ret = ""
        for i in range(self.kalmans.shape[0]):
            ret += "  {0}: {1:0<5.3f} * {2} {3}    kf.S: {4}\n".format(i, self.memp.weights[i],
                  self.memp.gausses[i], self.memp.particles[i], self.kalmans[i].S)
        return ret[:-1]  # trim the last newline

    def bayes(self, yt, cond = None):
        r"""Perform Bayes rule for new measurement :math:`y_t`. Uses following algorithm:

        1. generate new b parts of particles: :math:`b_t^{(i)} = \text{sample from }
           p(b_t^{(i)}|b_{t-1}^{(i)}) \quad \forall i`
        2. :math:`\text{set } Q_i := b_t^{(i)} \quad R_i := b_t^{(i)}` where :math:`Q_i, R_i` is
           covariance of process (respectively observation) noise in ith Kalman filter.
        3. perform Bayes rule for each Kalman filter using passed observation :math:`y_t`
        4. recompute weights: :math:`\omega_i = p(y_t | y_{1:t}, b_t^{(i)}) \omega_i` where
           :math:`p(y_t | y_{1:t}, b_t^{(i)})` is *evidence* (*marginal likehood*) pdf of ith Kalman
           filter.
        5. normalise weights
        6. resample particles
        """
        for i in range(self.kalmans.shape[0]):
            # generate new b_t
            self.memp.particles[i] = self.p_bt_btp.sample(self.memp.particles[i])

            # assign b_t to kalman filter
            # TODO: more general and correct apprach would be some kind of QRKalmanFilter that would
            # accept b_t in condition. This is planned in future.
            kalman = self.kalmans[i]
            kalman.Q[0,0] = self.memp.particles[i,0]
            kalman.R[0,0] = self.memp.particles[i,0]

            kalman.bayes(yt)

            self.memp.weights[i] *= exp(kalman.evidence_log(yt))

        # make sure that weights are normalised
        self.memp.normalise_weights()
        # resample particles
        self._resample()
        return True

    def _resample(self):
        indices = self.memp.get_resample_indices()
        self.kalmans = self.kalmans[indices]  # resample kalman filters (makes references, not hard copies)
        self.memp.particles = self.memp.particles[indices]  # resample particles
        for i in range(self.kalmans.shape[0]):
            if indices[i] == i:  # copy only when needed
                continue
            self.kalmans[i] = deepcopy(self.kalmans[i])  # we need to deep copy ith kalman
            self.memp.gausses[i] = self.kalmans[i].P  # reassign reference to correct (new) state pdf

        self.memp.weights[:] = 1./self.kalmans.shape[0]  # set weights to 1/n
        return True

    def posterior(self):
        return self.memp
