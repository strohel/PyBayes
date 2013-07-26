# Copyright (c) 2013 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for stresses"""

import numpy as np

import functools
import time


class Timer(object):
    """Simple timer used to measure real and cpu time of stresses."""

    def __init__(self):
        self.spent = "undefined", "undefined"

    def start(self):
        self.start_time = np.array([time.time(), time.clock()])

    def stop(self):
        self.spent = np.array([time.time(), time.clock()]) - self.start_time

    def __str__(self):
        return "Time spent: {0}s real time; {1}s CPU time".format(self.spent[0],
               self.spent[1])


def timed(func):
    """Decorator to mark a test as timed, provides timer argument"""
    @functools.wraps(func)
    def wrapper(self):
        timer = Timer()
        func(self, timer)
        print "{0}(): {1}".format(func.__name__, timer)
    return wrapper
