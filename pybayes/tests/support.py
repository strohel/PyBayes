# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for tests"""

import unittest as ut

import numpy as np

class PbTestCase(ut.TestCase):
    """Test case that adds some numeric assert functions"""

    def assertApproxEqual(self, X, Y):
        """Return true if X = Y to within machine precision

        Function for checking that different matrices from different
        computations are in some sense "equal" in the verification tests.
        """
        fuzz = 1.0e-8

        if np.all(abs(X - Y) < fuzz):
            return
        else:
            self.fail("NumPy arrays {0} and {1} are not fuzzy equal (+- {2})".format(X, Y, fuzz))
