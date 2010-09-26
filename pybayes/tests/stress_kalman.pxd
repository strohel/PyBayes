# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import cython  # for decorators
cimport numpy as np

cimport pybayes.kalman as kf
cimport pybayes.pdfs as pdfs
cimport pybayes.utils as utils


#@cython.boundscheck(False)
#@cython.wraparound(False)
@cython.locals(y = "np.ndarray[utils.data_t, ndim=2]",
               u = "np.ndarray[utils.data_t, ndim=2]",
               Mu_py = "np.ndarray[utils.data_t, ndim=2]",
               kalman = kf.Kalman,
               t = int)
cpdef run_kalman_on_mat_data(input_file, output_file)
