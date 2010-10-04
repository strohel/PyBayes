# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Various support methods for tests"""

from numpy import sum


def approx_eq(X, Y):
    """Return true if X = Y to withing machine precision

    Function for checking that different matrices from different
    computations are some sense "equal" in the verification tests.
    """
    return abs(sum(X - Y)) < 1e-5
