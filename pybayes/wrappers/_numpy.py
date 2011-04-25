# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Wrapper around numpy - python version"""

from numpy import *

#from numpy import any as np_any, arange, array, asarray, cumsum, diag, dot, dot as dotvv, empty, exp, ndarray, ones, prod, sum, zeros
#from numpy.linalg import cholesky, inv
#import numpy.random as random


## support NumPy before 1.5.0 by emulating its slogdet
#try:
    #numpy.linalg.slogdet
#except AttributeError:
    #from math import log

    #def _slogdet(a):
        #d = numpy.linalg.det(a)
        #if d == 0:
            #return (0., float('-inf'))
        #if d > 0:
            #return (1., log(d))
        #else:
            #return (-1., log(-d))

    #numpy.linalg.slogdet = _slogdet

dotvv = dot  # dot vector*vector is defined as dot in python version
