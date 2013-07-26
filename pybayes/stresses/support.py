# Copyright (c) 2013 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for stresses"""

import numpy as np

import functools
import time
# Python 2.6 compatibility
try:
    from unittest import skip, skipIf, skipUnless
except ImportError:
    skip = None


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


if skip is None:
    def skip(reason):
        """Implementation of the @skip decorator from Python 2.7 for Python 2.6"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                pass
            origdoc = wrapper.__doc__ or wrapper.__name__
            wrapper.__doc__ = wrapper.__name__ + " [skipped '{0}']".format(reason)
            return wrapper
        return decorator

    def _id(obj):
            return obj

    def skipIf(condition, reason):
        """Implementation of the @skipIf decorator from Python 2.7 for Python 2.6"""
        if condition:
            return skip(reason)
        return _id

    def skipUnless(condition, reason):
        """Implementation of the @skipUnless decorator from Python 2.7 for Python 2.6"""
        if not condition:
            return skip(reason)
        return _id
