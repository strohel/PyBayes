#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Run PyBayes' stress-suite"""

import cProfile
import pstats
from optparse import OptionParser
import os

import pybayes.tests.stress_kalman


# parse cmdline arguments
parser = OptionParser(description="Run PyBayes stress suite with user-supplied data")
parser.add_option("-p", "--profile", action="store_true", dest="profile",
                  help="run stresses under profiler (default: no)", default=False)
parser.add_option("-d", "--datadir", action="store", dest="datadir",
                  help="directory containing stress data", default="stress_data")
(options, args) = parser.parse_args()

if not os.path.isdir(options.datadir):
    print("Error: supplied (or default) datadir '{0}' is not a directory!".format(options.datadir))
    parser.print_help()
    exit(1)

# define stress tests
stresses = [pybayes.tests.stress_kalman]

# run stress tests
for stress in stresses:
    name = stress.__name__
    print(name + ":")
    if options.profile:
        filename = "profile_" + name + ".prof"

        cProfile.runctx(name + ".main(options)", globals(), locals(), filename)

        s = pstats.Stats(filename)
        s.sort_stats("time").print_stats()
    else:
        stress.main(options)
