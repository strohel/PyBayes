#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Run PyBayes' stress-suite"""

import pstats, cProfile

import pybayes.tests.stress_kalman

stresses = [pybayes.tests.stress_kalman]

for stress in stresses:
    name = stress.__name__
    filename = "profile_" + name + ".prof"

    print(name + ":")
    cProfile.runctx(name + ".main()", globals(), locals(), filename)

    s = pstats.Stats(filename)
    s.sort_stats("cumulative").print_stats()
