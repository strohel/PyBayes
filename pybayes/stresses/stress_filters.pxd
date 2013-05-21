# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython
cimport numpy as np

cimport pybayes as pb


#@cython.boundscheck(False)
#@cython.wraparound(False)
@cython.locals(kalman = pb.KalmanFilter,
               y = np.ndarray,
               u = np.ndarray,
               mean = np.ndarray,
               t = int,
               N = int)
cpdef run_kalman_on_mat_data(input_file, output_file, timer)

@cython.locals(nr_steps = int,
               pf = pb.Filter)
cpdef run_pf(timer, pf_opts, nr_particles, pf_class)
