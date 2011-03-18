# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Stresses for kalman filters"""

import os.path
import time

import numpy as np
from scipy.io import loadmat, savemat

import pybayes as pb


def stress_kalman(options, timer):
    input_file = options.datadir + "/stress_kalman_data.mat"
    output_file = "stress_kalman_res.mat"

    run_kalman_on_mat_data(input_file, output_file, timer)

def run_kalman_on_mat_data(input_file, output_file, timer):
    d = loadmat(input_file, struct_as_record=True, mat_dtype=True)

    mu0 = np.reshape(d.pop('mu0'), (-1,))  # otherwise we would get 2D array of shape (1xN)
    P0 = d.pop('P0')
    y = d.pop('y').T
    u = d.pop('u').T

    gauss = pb.GaussPdf(mu0, P0)
    kalman = pb.KalmanFilter(d['A'], d['B'], d['C'], d['D'], d['Q'], d['R'], gauss)

    N = y.shape[0]
    n = mu0.shape[0]
    Mu_py = np.zeros((N, n))

    timer.start()
    for t in xrange(1, N):  # the 1 start offset is intentional
        Mu_py[t] = kalman.bayes(y[t], u[t]).mu
    timer.stop()

    Mu_py = Mu_py.T
    savemat(output_file, {"Mu_py":Mu_py, "exec_time_pybayes":timer.spent[0]}, oned_as='row')

def stress_pf_1(options, timer):
    raise Exception("Stress skipped")
    nr_particles = 100  # number of particles
    nr_steps = 50 # number of time steps

    # prepare random vector components:
    a_t, b_t = pb.RVComp(1, 'a_t'), pb.RVComp(1, 'b_t')  # state in t
    a_tp, b_tp = pb.RVComp(1, 'a_{t-1}'), pb.RVComp(1, 'b_{t-1}')  # state in t-1

    # prepare callback functions
    def f(x):  # take a_t out of [a_t, b_t]
        return x[0:1]
    def g(x):  # exponential of b_t out of [a_t, b_t]
        return np.exp(x[1:2])
    #def g(x):  # take b_t out of [a_t, b_t]
        #return x[1:2]

    # prepare p(x_t | x_{t-1}) density:
    #p1 = pb.GaussCPdf(1, 2, f, g, pb.RV(a_t), pb.RV(a_tp, b_t))
    p1 = pb.LinGaussCPdf(1., 0., 1., 0., pb.RV(a_t), pb.RV(a_tp, b_t))
    cov, A, b = np.array([[0.0001]]), np.array([[1.]]), np.array([0.])  # params for p2
    p2 = pb.MLinGaussCPdf(cov, A, b, pb.RV(b_t), pb.RV(b_tp))
    p_xt_xtp = pb.ProdCPdf((p1, p2), pb.RV(a_t, b_t), pb.RV(a_tp, b_tp))

    # prepare p(y_t | x_t) density:
    p_yt_xt = pb.GaussCPdf(1, 2, f, g)

    # construct initial particle density and particle filter:
    init_pdf = pb.UniPdf(np.array([-0.01, 0.08]), np.array([0.01, 0.12]))
    pf = pb.ParticleFilter(nr_particles, init_pdf, p_xt_xtp, p_yt_xt)

    x_t = np.array([0., 0.])
    y_t = np.empty(1)
    timer.start()
    for i in range(nr_steps):
        x_t[1] = (i+10.)/100.

        # simulate random process:
        x_t[0:1] = p1.sample(x_t)  # this is effectively [a_{t-1}, b_t]
        print "simulated x_{0} = {1}".format(i, x_t)
        y_t = p_yt_xt.sample(x_t)
        #y_t[0] = x_t[0]
        print "simulated y_{0} = {1}".format(i, y_t)

        #print pf.emp.particles
        apost = pf.bayes(y_t)
        print "returned mean = {0}".format(apost.mean())
        print
    timer.stop()

def stress_pf_2(options, timer):
    nr_particles = 100  # number of particles
    nr_steps = 50 # number of time steps

    # prepare random vector components:
    a_t, b_t = pb.RVComp(1, 'a_t'), pb.RVComp(1, 'b_t')  # state in t
    a_tp, b_tp = pb.RVComp(1, 'a_{t-1}'), pb.RVComp(1, 'b_{t-1}')  # state in t-1

    # prepare callback functions
    def f(x):  # take a_t out of [a_t, b_t]
        return x[0:1]
    def g(x):  # exponential of b_t out of [a_t, b_t]
        return np.exp(x[1:2])
    #def g(x):  # take b_t out of [a_t, b_t]
        #return x[1:2]

    # prepare p(x_t | x_{t-1}) density:
    cov, A, b = np.array([[1.]]), np.array([[1.]]), np.array([0.])  # params for p1
    p1 = pb.MLinGaussCPdf(cov, A, b, pb.RV(a_t), pb.RV(a_tp))
    p2 = pb.LinGaussCPdf(1., 0., 1., 0., pb.RV(b_t), pb.RV(b_tp, a_tp))
    p_xt_xtp = pb.ProdCPdf((p1, p2), pb.RV(a_t, b_t), pb.RV(a_tp, b_tp))

    # prepare p(y_t | x_t) density:
    p_yt_xt = pb.GaussCPdf(1, 2, f, g)

    # construct initial particle density and particle filter:
    init_pdf = pb.UniPdf(np.array([2., -10.]), np.array([3., -3.]))
    pf = pb.ParticleFilter(nr_particles, init_pdf, p_xt_xtp, p_yt_xt)

    x_t = np.array([0., -10.])
    y_t = np.empty(1)
    timer.start()
    for i in range(nr_steps):
        x_t[0] = 2.5 + i/50.  # set a_t

        # simulate random process:
        x_t[1:2] = p2.sample(x_t[np.array([1, 0])])  # this is effectively b_t = sample from p [b_{t-1}, a_{t-1}]
        print "simulated x_{0} = {1}".format(i, x_t)
        y_t = p_yt_xt.sample(x_t)
        print "simulated y_{0} = {1}".format(i, y_t)

        #print pf.emp.particles
        apost = pf.bayes(y_t)
        print "returned mean = {0}".format(apost.mean())
        print
    timer.stop()
