# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - python version"""

# just import and flatten numpy types and functions

from numpy import any as np_any, array, asarray, diag, dot, dot as dotvv, ndarray, zeros
from numpy.linalg import cholesky, inv, slogdet
from numpy.random import normal, uniform
