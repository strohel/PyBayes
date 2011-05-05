# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython
cimport numpy as np

cimport pybayes as pb


#@cython.boundscheck(False)
#@cython.wraparound(False)
@cython.locals(kalman = pb.ControlKalmanFilter,
               y = np.ndarray,
               u = np.ndarray,
               Mu_py = np.ndarray,
               t = int)
cpdef run_kalman_on_mat_data(input_file, output_file, timer)

@cython.locals(nr_steps = int,
               pf = pb.Filter,
               apost = pb.Pdf)
cpdef run_pf(options, timer, pf_opts, nr_particles, pf_class)
