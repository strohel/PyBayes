# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# wrappers.numpy test needs this file, because in cython wrappers.numpy, there are only
# cdefs, not cpdefs

cimport pybayes.wrappers._numpy as nw
