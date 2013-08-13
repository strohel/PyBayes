# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for tests"""

import functools
import numpy as np
import sys
import unittest as ut
try:
    from unittest.case import _ExpectedFailure as ExpectedFailure
except ImportError:
    ExpectedFailure = None


def stochastic(func):
    """Decorator to mark test as stochastic - these tests are not fatal when they fail."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ut.TestCase.failureException:
            if ExpectedFailure is not None:  # added in Py 2.7
                raise ExpectedFailure(sys.exc_info())
    wrapper.__doc__ += ' (stochastic, failures ignored)'
    return wrapper


class PbTestCase(ut.TestCase):
    """Test case that adds some numeric assert functions"""

    def assertApproxEqual(self, X, Y):
        """Return true if X = Y to within machine precision

        Function for checking that different matrices from different
        computations are in some sense "equal" in the verification tests.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        fuzz = 1.0e-8

        self.assertEqual(X.ndim, Y.ndim)
        self.assertEqual(X.shape, Y.shape)

        if np.all(X == Y):  # catches -inf == -inf etc.
            return
        if np.all(abs(X - Y) < fuzz):
            return
        self.fail("NumPy arrays {0} and {1} are not fuzzy equal (+- {2})".format(X, Y, fuzz))

    def assertArraysEqualNotSame(self, a, b):
        """Assert that numpy arrays a and b are equal, but are not the same instances"""
        self.assertNotEqual(id(a), id(b))
        self.assertApproxEqual(a, b)

    def assertArraysSame(self, a, b):
        # id(a) == id(b) doen't work for Cython memoryview arrays
        self.assertEqual(a.ndim, b.ndim)
        self.assertEqual(a.shape, b.shape)
        if a.ndim == 1:
            if a.shape[0] == 0:
                return  # hard to test in this case
            orig_a, orig_b = a[0], b[0]
            a[0] = b[0] = 0  # reset
            a[0] = 1  # does it propagate to b?
            self.assertEqual(a[0], b[0])
            a[0], b[0] = orig_a, orig_b  # set back
        elif a.ndim == 2:
            if a.shape[0] == 0 or a.shape[1] == 0:
                return  # hard to test
            orig_a, orig_b = a[0, 0], b[0, 0]
            a[0, 0] = b[0, 0] = 0  # reset
            a[0, 0] = 1  # does it propagate to b?
            self.assertEqual(a[0, 0], b[0, 0])
            a[0, 0], b[0, 0] = orig_a, orig_b  # set back
        else:
            self.fail("More than 2 or less than 1 dimensions not supported in this method")

    def assertRVsEqualNotSame(self, a, b):
        """Assert that :class:`~pybayes.pdfs.RV` objects a and b are equal, but
        are not the same instances neither shallow copies of themselves.

        RVs are special case during deepcopy - the RVComps should be referenced,
        not copied."""
        self.assertNotEqual(id(a), id(b))
        self.assertNotEqual(id(a.components), id(b.components))
        self.assertEqual(a.name, b.name)  # no need to test for id inequality - strings are immutable
        self.assertEqual(a.dimension, b.dimension)  # ditto
        for (a_comp, b_comp) in zip(a.components, b.components):
            # equality for rv comps is defined as object instance identity
            self.assertEqual(a_comp, b_comp)
