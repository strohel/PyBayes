#!/usr/bin/env python
# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org/
#
# Contributed by Sebastien Loisel
# Fixed by Isaac Gouy
# Sped up by Josh Goldfoot
# Dirtily sped up by Simon Descarpentries
# Sped up with numpy by Kittipong Piyawanno
# 2to3

from sys import argv
from numpy import *

def spectralnorm(n):
    u = matrix(ones(n))
    j = arange(n)
    eval_func = lambda i : 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1)
    M = matrix([eval_func(i) for i in arange(n)])
    MT = M.T
    for i in range (10):
        v = (u*MT)*M
        u = (v*MT)*M
    print("%0.9f" % (sum(u*v.T)/sum(v*v.T))**0.5)

spectralnorm(int(argv[1]))
