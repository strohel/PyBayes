#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.core import setup

try:
    from Cython.Distutils import build_ext
    from Cython.Distutils.extension import Extension
except ImportError:
    cython = False
else:
    cython = True


params = {'name':'PyBayes',
          'version':'0.1-pre',
          'description':'Library for convenient and fast Bayesian decision making',
          'author':'Matej Laitl',
          'author_email':'matej@laitl.cz',
          'url':'http://github.com/strohel/PyBayes',
          'packages':['pybayes', 'pybayes.tests']  # defined even for cython version -
          # we use this to include __init__.py's and to provide fallback
         }

if cython:
    # determine path to NumPy C header files
    try:
        import numpy
    except ImportError:
        print("Error: Cannot import numpy. It is needed at build-time in order to determine")
        print("       include path for it")
        numpy_path = None
        exit()
    else:
        numpy_path = numpy.__path__
        del numpy

    incl = [element + "/core/include" for element in numpy_path]
    params['cmdclass'] = {'build_ext': build_ext}
    params['ext_package'] = 'pybayes'
    params['ext_modules'] = [Extension('pdfs', ['pybayes/pdfs.py'], include_dirs=incl),
                             Extension('kalman', ['pybayes/kalman.py'], include_dirs=incl),
                             Extension('tests.test_pdfs', ['pybayes/tests/test_pdfs.py'], include_dirs=incl),
                             Extension('tests.test_kalman', ['pybayes/tests/test_kalman.py'], include_dirs=incl),
                            ]
else:
    print("Warning: cython was not found on your system. Falling back to pure")
    print("         python mode which is significantly slower.")

setup(**params)
