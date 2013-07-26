# Copyright (c) 2011 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""PyBayes test-suite runner. Used when user calls `python -m pybayes.tests"""

import unittest as ut

from pybayes.tests import *


if __name__ == '__main__':
    ut.main()
