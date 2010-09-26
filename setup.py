#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import os.path
import sys

from distutils.core import setup


# naive option parsing (cannot conveniently use optparse or getopt)
use_cython = "untouched"
argv = sys.argv[:]  # make copy to be safe
for i in range(1, len(argv)):
    if argv[i].startswith("--use-cython="):
        if argv[i][13:] == "yes": use_cython = True
        elif argv[i][13:] == "no": use_cython = False
        elif argv[i][13:] == "auto": use_cython = None  # autodetect
        else:
            print("Error: Argument to --use-cython muse be one of yes, no or auto.")
            exit(1)
        del sys.argv[i]  # do not confuse distutils parsing
if use_cython not in (True, False, None):
    use_cython = None  # autodetect
    print("Notice: Assuming --use-cython=auto. To override, pas --use-cython={yes,no,auto}.")

# configure build
if use_cython is not False:
    # autodetect (or check for) cython
    try:
        from Cython.Distutils import build_ext
        from Cython.Distutils.extension import Extension
    except ImportError:
        if use_cython is True:
            print("Error: cython was not found and --use-cython=yes was passed.")
            print("       please install cython in order to build faster PyBayes.")
        else:  # use_cython is None (autodetect)
            print("Warning: cython was not found on your system. Falling back to pure")
            print("         python mode which is significantly slower.")
        use_cython = False
    else:
        use_cython = True

params = {'name':'PyBayes',
          'version':'0.1-pre',
          'description':'Library for convenient and fast Bayesian decision making',
          'author':'Matej Laitl',
          'author_email':'matej@laitl.cz',
          'url':'http://github.com/strohel/PyBayes',
          #'package_data':{'pybayes.tests':['stress_kalman_data.mat']}  # this unfortunately
          # breaks cython build, as params['packages'] is empty then
         }

if use_cython is True:
    # determine path to NumPy C header files
    try:
        import numpy
    except ImportError:
        print("Error: Cannot import numpy. It is needed at build-time in order to determine")
        print("       include path for it. NumPy is needed runtime for every PyBayes build")
        print("       and buid-time for cython build.")
        numpy_path = None
        exit()
    else:
        numpy_path = numpy.__path__
        del numpy

    params['cmdclass'] = {'build_ext': build_ext}
    params['py_modules'] = ['pybayes.__init__', 'pybayes.tests.__init__']

    extensions = ['kalman.py',
                  'pdfs.py',
                  'utils.py',

                  'tests/stress_kalman.pyx',
                  'tests/test_kalman.py',
                  'tests/test_pdfs.py',
                 ]
    # add numpy directory so that included .h files can be found
    incl = [element + "/core/include" for element in numpy_path]
    compile_args=["-O2"]
    link_args=["-Wl,-O1"]
    params['ext_modules'] = []
    for extension in extensions:
        module = "pybayes." + os.path.splitext(extension)[0].replace("/", ".")
        paths = ["pybayes/" + extension]
        params['ext_modules'].append(
            Extension(module, paths, include_dirs=incl, extra_compile_args=compile_args,
                      extra_link_args=link_args)
        )

else:  # use_cython is False
    params['packages'] = ['pybayes', 'pybayes.tests']

setup(**params)
