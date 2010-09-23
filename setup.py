#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.core import setup

setup(name='PyBayes',
      version='0.1-pre',
      description='Library for convenient and fast Bayesian decision making',
      author='Matej Laitl',
      author_email='matej@laitl.cz',
      url='http://github.com/strohel/PyBayes',
      packages=['pybayes', 'pybayes.tests'],
     )
