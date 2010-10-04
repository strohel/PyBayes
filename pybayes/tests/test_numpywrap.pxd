# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# numpywrap test needs this file, because in cython numpywrap, there are only
# cdefs, not cpdefs

cimport pybayes.numpywrap as nw
