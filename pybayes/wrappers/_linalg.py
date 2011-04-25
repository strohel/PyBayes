# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy.linalg - python version"""

from numpy.linalg import *

# support NumPy before 1.5.0 by emulating its slogdet
try:
    slogdet
except NameError:
    from math import log

    def slogdet(a):
        d = numpy.linalg.det(a)
        if d == 0:
            return (0., float('-inf'))
        if d > 0:
            return (1., log(d))
        else:
            return (-1., log(-d))
