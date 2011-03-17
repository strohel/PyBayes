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
    nr_particles = 100  # number of particles
    N = 100 # number of time steps

    # prepare pdfs:
    a_t, b_t = pb.RVComp(1, 'a_t'), pb.RVComp(1, 'b_t')  # state in t
    a_tp, b_tp = pb.RVComp(1, 'a_{t-1}'), pb.RVComp(1, 'b_{t-1}')  # state in t-1
    p1 = pb.LinGaussCPdf(1., 0., 1., 0., pb.RV(a_t), pb.RV(a_tp, b_t))
    cov, A, b = np.array([[0.0001]]), np.array([[1.]]), np.array([0.])  # params for p2
    p2 = pb.MLinGaussCPdf(cov, A, b, pb.RV(b_t), pb.RV(b_tp))
    p_xt_xtp = pb.ProdCPdf((p1, p2), pb.RV(a_t, b_t), pb.RV(a_tp, b_tp))

    y_t = pb.RVComp(1, "y_t")  # observation in t
    p1 = pb.MLinGaussCPdf(np.array([[1.]]), np.array([[1.]]), np.array([0.]), pb.RV(y_t), pb.RV(b_t))  # TODO: jine nez v zadani
    p_yt_xt = pb.ProdCPdf((p1,), pb.RV(y_t), pb.RV(a_t, b_t))

    init_pdf = pb.UniPdf(np.array([3., 3.]), np.array([5., 5.]))
    pf = pb.ParticleFilter(nr_particles, init_pdf, p_xt_xtp, p_yt_xt)

    timer.start()
    for i in range(N):
        pf.bayes(np.random.uniform(0.5, 3., (1,)))
    timer.stop()
