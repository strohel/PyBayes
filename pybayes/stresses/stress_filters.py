# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Stresses for kalman filters"""

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import scipy.io
except ImportError:
    scipy = None

from os.path import dirname, join
import unittest as ut

import pybayes as pb
from support import timed


def run_kalman_on_mat_data(input_file, output_file, timer):
    # this should be here so that only this stress fails when scipy is not installed
    loadmat = scipy.io.loadmat
    savemat = scipy.io.savemat

    d = loadmat(input_file, struct_as_record=True, mat_dtype=True)

    mu0 = np.reshape(d.pop('mu0'), (-1,))  # otherwise we would get 2D array of shape (1xN)
    P0 = d.pop('P0')
    y = d.pop('y').T
    u = d.pop('u').T
    #x = d.pop('x').T
    x = None

    gauss = pb.GaussPdf(mu0, P0)
    kalman = pb.KalmanFilter(d['A'], d['B'], d['C'], d['D'], d['Q'], d['R'], gauss)

    N = y.shape[0]
    n = mu0.shape[0]
    mean = np.zeros((N, n))
    var = np.zeros((N, n))

    timer.start()
    for t in xrange(1, N):  # the 1 start offset is intentional
        kalman.bayes(y[t], u[t])
        mean[t] = kalman.posterior().mean()
        #var[t]  = kalman.posterior().variance()
    timer.stop()

    var = np.sqrt(var)  # to get standard deviation
    plt = None  # turn off plotting for now
    if plt:
        axis = np.arange(N)
        plt.plot(axis, x[:,0], 'k-', label='x_1')
        plt.plot(axis, x[:,1], 'k--', label='x_2')
        plt.errorbar(axis, mean[:,0], fmt='s', label='mu_1')  # yerror=var[:,0]
        plt.errorbar(axis, mean[:,1], fmt='D', label='mu_2')  # yerror=var[:,1]
        plt.legend()
        plt.show()

    savemat(output_file, {"Mu_py":mean.T, "exec_time_pybayes":timer.spent[0]}, oned_as='row')


class PfOptionsA(object):
    """Class that represents options for a particle filter"""

    def __init__(self, nr_steps):
        print "Generating random data for particle filter stresses A..."
        self.nr_steps = nr_steps

        # prepare random variable components:
        a_t, b_t = pb.RVComp(1, 'a_t'), pb.RVComp(1, 'b_t')  # state in t
        a_tp, b_tp = pb.RVComp(1, 'a_{t-1}'), pb.RVComp(1, 'b_{t-1}')  # state in t-1

        # arguments to Kalman filter part of the marginalized particle filter
        self.kalman_args = {}

        # prepare callback functions
        sigma_sq = np.array([0.0001])
        def f(cond):  # log(b_{t-1}) - 1/2 \sigma^2
            return np.log(cond) - sigma_sq/2.
        def g(cond):  # \sigma^2
            return sigma_sq

        # p(a_t | a_{t-1} b_t) density:
        p_at_atpbt = pb.LinGaussCPdf(1., 0., 1., 0., [a_t], [a_tp, b_t])
        self.kalman_args['A'] = np.array([[1.]])  # process model
        # p(b_t | b_{t-1}) density:
        self.p_bt_btp = pb.GaussCPdf(1, 1, f, g, rv=[b_t], cond_rv=[b_tp], base_class=pb.LogNormPdf)
        # p(x_t | x_{t-1}) density:
        self.p_xt_xtp = pb.ProdCPdf((p_at_atpbt, self.p_bt_btp), [a_t, b_t], [a_tp, b_tp])

        # prepare p(y_t | x_t) density:
        self.p_yt_xt = pb.LinGaussCPdf(1., 0., 1., 0.)
        self.kalman_args['C'] = np.array([[1.]])  # observation model

        # Initial [a_t, b_t] from .. to:
        self.init_range = np.array([[-18., 0.3], [-14., 0.7]])
        init_mean = (self.init_range[0] + self.init_range[1])/2.

        x_t = np.zeros((nr_steps, 2))
        x_t[0] = init_mean.copy()
        y_t = np.empty((nr_steps, 1))
        for i in range(nr_steps):
            # set b_t:
            x_t[i,1] = i/100. + init_mean[1]
            # simulate random process:
            x_t[i,0:1] = p_at_atpbt.sample(x_t[i])  # this is effectively [a_{t-1}, b_t]
            y_t[i] = self.p_yt_xt.sample(x_t[i])
            # DEBUG: print "simulated x_{0} = {1}".format(i, x_t[i])
            # DEBUG: print "simulated y_{0} = {1}".format(i, y_t[i])
        self.x_t = x_t
        self.y_t = y_t


def run_pf(timer, pf_opts, nr_particles, pf_class):
    nr_steps = pf_opts.nr_steps # number of time steps

    # prepare initial particle density:
    init_pdf = pb.UniPdf(pf_opts.init_range[0], pf_opts.init_range[1])

    # construct particle filter
    if pf_class is pb.ParticleFilter:
        pf = pf_class(nr_particles, init_pdf, pf_opts.p_xt_xtp, pf_opts.p_yt_xt)
    elif pf_class is pb.MarginalizedParticleFilter:
        pf = pf_class(nr_particles, init_pdf, pf_opts.p_bt_btp, pf_opts.kalman_args)
    else:
        raise NotImplementedError("This switch case not handled")

    x_t = pf_opts.x_t
    y_t = pf_opts.y_t
    mean = np.empty((nr_steps, 2))

    timer.start()
    for i in range(nr_steps):
        pf.bayes(y_t[i])
        mean[i] = pf.posterior().mean()
    timer.stop()
    cumerror = np.sum((mean - x_t)**2, 0)
    print "  {0}-{3} cummulative error for {1} steps: {2}".format(
        nr_particles, nr_steps, np.sqrt(cumerror), pf_class.__name__)
    plt = None  # disable plotting for now
    if plt:
        x = np.arange(nr_steps)
        plt.plot(x, mean[:,0], 'x', label="{0}: {1}".format(nr_particles, pf_class.__name__))
        plt.plot(x, mean[:,1], '+', label="{0}: {1}".format(nr_particles, pf_class.__name__))
    if plt and nr_particles == 90 and pf_class == pb.MarginalizedParticleFilter:
        plt.plot(x, x_t[:,0], '-')
        plt.plot(x, x_t[:,1], '--')
        plt.legend()
        plt.show()


class StressFilters(ut.TestCase):
    pf_nr_steps = 100  # number of steps for particle filter
    pf_opts_a = PfOptionsA(pf_nr_steps)

    @ut.skipUnless(scipy, "Kalman stress needs SciPy installed")
    @timed
    def test_kalman(self, timer):
        input_file = join(dirname(__file__), "data", "stress_kalman_data.mat")
        output_file = "stress_kalman_res.mat"
        run_kalman_on_mat_data(input_file, output_file, timer)

    @timed
    def test_pf_a_1(self, timer):
        run_pf(timer, self.pf_opts_a, 10, pb.ParticleFilter)

    @timed
    def test_pf_a_1_marg(self, timer):
        run_pf(timer, self.pf_opts_a, 10, pb.MarginalizedParticleFilter)

    @timed
    def test_pf_a_2(self, timer):
        run_pf(timer, self.pf_opts_a, 30, pb.ParticleFilter)

    @timed
    def test_pf_a_2_marg(self, timer):
        run_pf(timer, self.pf_opts_a, 30, pb.MarginalizedParticleFilter)

    @timed
    def test_pf_a_3(self, timer):
        run_pf(timer, self.pf_opts_a, 90, pb.ParticleFilter)

    @timed
    def test_pf_a_3_marg(self, timer):
        run_pf(timer, self.pf_opts_a, 90, pb.MarginalizedParticleFilter)
