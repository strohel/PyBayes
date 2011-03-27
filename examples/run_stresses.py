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
        self.cumtime = 0.  # cummulative time

    def start(self):
        self.start_time = np.array([time.time(), time.clock()])

    def stop(self):
        self.spent = np.array([time.time(), time.clock()]) - self.start_time
        self.cumtime += self.spent[0]

    def __str__(self):
        return "Time spent: {0}s real time; {1}s CPU time".format(self.spent[0],
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

def a_function():
    pass

def scan_for_stresses():
    ret = []
    globs = globals()
    functype = type(a_function)
    builtinfunctype = type(dir)  # needed for compiled cython code
    for key in globs:
        if not key.startswith("stress_"):
            continue
        value = globs[key]
        if type(value) in (functype, builtinfunctype):
            ret.append(value)
    def keyfunc(obj):
        return obj.__name__
    ret.sort(key=keyfunc)
    return ret

stresses = scan_for_stresses()

timer = Timer()

# run stress tests
count = 0
failed = 0
for stress in stresses:
    name = stress.__name__
    print name + "():"
    if options.profile:
        filename = "profile_" + name + ".prof"

        cProfile.runctx(name + "(options, timer)", globals(), locals(), filename)

        s = pstats.Stats(filename)
        s.sort_stats("cumulative").print_stats()  # or sort_stats("time")
    else:
        try:
            stress(options, timer)
        except Exception, e:
            print "  Exception occured:", e
            failed += 1
        else:
            print "  {0}".format(timer)
    count += 1

print "Ran {0} stresses in {1}s, {2} of them failed.".format(
      count, timer.cumtime, failed)
print
