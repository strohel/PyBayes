#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Run PyBayes' stress-suite"""

import cProfile
import pstats
from optparse import OptionParser
import os

import numpy as np

from pybayes.stresses import *


class Timer(object):
    """Simple timer used to measure real and cpu time of stresses."""

    def __init__(self):
        self.cummulative = 0.

    def start(self):
        self.start = np.array([time.time(), time.clock()])

    def stop(self):
        self.spent = np.array([time.time(), time.clock()]) - self.start
        self.cummulative += self.spent[0]

    def __str__(self):
        return "time spent: {0}s real time; {1}s CPU time".format(self.spent[0],
               self.spent[1])


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
stresses = [stress_kalman, stress_pf_1]

timer = Timer()

# run stress tests
count = 0
failed = 0
for stress in stresses:
    name = stress.__name__
    print name + ":"
    if options.profile:
        filename = "profile_" + name + ".prof"

        cProfile.runctx(name + "(options, timer)", globals(), locals(), filename)

        s = pstats.Stats(filename)
        s.sort_stats("time").print_stats()
    else:
        try:
            stress(options, timer)
        except Exception, e:
            print "   Exception occured:", e
            failed += 1
        else:
            print "   {0}".format(timer)
    count += 1

print "Ran", count, "stresses,", failed, "of them failed."
print
