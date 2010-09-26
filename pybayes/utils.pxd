# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport numpy as np


ctypedef np.float64_t data_t

cpdef np.ndarray dot(np.ndarray a, np.ndarray b)
