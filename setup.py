#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.core import setup, Extension

# TODO: enclose in try/catch
from Cython.Distutils import build_ext

cython = True

params = {'name':'PyBayes',
          'version':'0.1-pre',
          'description':'Library for convenient and fast Bayesian decision making',
          'author':'Matej Laitl',
          'author_email':'matej@laitl.cz',
          'url':'http://github.com/strohel/PyBayes',
         }

if cython:
    params['cmdclass'] = {'build_ext': build_ext}
    params['ext_package'] = 'pybayes'
    params['ext_modules'] = [Extension('pdfs', ['pybayes/pdfs.py']),
                             Extension('kalman', ['pybayes/kalman.py']),
                             Extension('tests.test_pdfs', ['pybayes/tests/test_pdfs.py']),
                             Extension('tests.test_kalman', ['pybayes/tests/test_kalman.py']),
                            ]
else:
    print("Warning: cython was not found on your system. Falling back to pure")
    print("         python mode which is significantly slower.")
    params['packages'] = ['pybayes', 'pybayes.tests']


setup(**params)
