#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Run PyBayes' stress-suite"""

import cProfile
import pstats
from optparse import OptionParser

import pybayes.tests.stress_kalman


# parse cmdline arguments
parser = OptionParser()
parser.add_option("-p", "--profile", action="store_true", dest="profile",
                  help="run stresses under profiler (default: no)", default=False)
(options, args) = parser.parse_args()

# define stress tests
stresses = [pybayes.tests.stress_kalman]

# run stress tests
for stress in stresses:
    name = stress.__name__
    print(name + ":")
    if options.profile:
        filename = "profile_" + name + ".prof"

        cProfile.runctx(name + ".main()", globals(), locals(), filename)

        s = pstats.Stats(filename)
        s.sort_stats("cumulative").print_stats()
    else:
        stress.main()
