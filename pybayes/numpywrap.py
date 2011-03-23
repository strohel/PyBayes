# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - python version"""

# just import and flatten numpy types and functions

from numpy import any as np_any, arange, array, asarray, cumsum, diag, dot, dot as dotvv, empty, exp, ndarray, ones, prod, sum, zeros
from numpy.linalg import cholesky, inv
import numpy.random as random

# support NumPy before 1.5.0 by emulating its slogdet
try:
    from numpy.linalg import slogdet
except ImportError:
    from numpy.linalg import det
    from math import log

    def slogdet(a):
        d = det(a)
        if d == 0:
            return (0, float('-inf'))
        if d > 0:
            return (1, log(d))
        else:
            return (-1, log(-d))
