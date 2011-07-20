#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import os.path
import sys

from distutils.core import setup


# generic distutils parameters
version = '0.2'
params = {
    # meta-data; see http://docs.python.org/distutils/setupscript.html#additional-meta-data
    'name':'PyBayes',
    'version':version,
    'author':u'Matěj Laitl',
    'author_email':'matej@laitl.cz',
    'maintainer':u'Matěj Laitl',
    'maintainer_email':'matej@laitl.cz',
    'url':'https://github.com/strohel/PyBayes',
    'description':'Python library for recursive Bayesian estimation (Bayesian filtering)',
    'long_description':'PyBayes is an object-oriented Python library for recursive Bayesian ' +
        'estimation (Bayesian filtering) that is convenient to use. Already implemented are ' +
        'Kalman filter, particle filter and marginalized particle filter, all built atop of ' +
        'a light framework of probability density functions. PyBayes can optionally use Cython ' +
        'for lage speed gains (Cython build is several times faster).',
    # Note to myself: must manually upload on each release!
    'download_url':'https://github.com/downloads/strohel/PyBayes/PyBayes-v'+version+'.tar.gz',
    'platforms':'cross-platform',
    'license':'GNU GPL v2+',
    'classifiers':[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]

    #'package_data':{'pybayes.tests':['stress_kalman_data.mat']}  # this unfortunately
    # does not work in cython build, as params['packages'] is empty then
}


# building code starts here

class Options(object):
    def __init__(self):
        self.use_cython = None
        self.profile = None
        self.numpy_include_dir = None
        self.build_ext = None
        self.Extension = None

def parse_cmdline_options():
    """
    Parse additional (extra) options passed to setup.py

    Returns
    -------
    options : object with use_cython (tristate), profile (bool) attributes.
              tristate options have values True, False and None which means
              autodetect.
    """
    options = Options()
    options.use_cython = "untouched"
    options.profile = "untouched"

    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--use-cython="):
            if sys.argv[i][13:] == "yes": options.use_cython = True
            elif sys.argv[i][13:] == "no": options.use_cython = False
            elif sys.argv[i][13:] == "auto": options.use_cython = None
            else:
                print("Error: Argument to --use-cython muse be one of yes, no or auto")
                exit(1)
            del sys.argv[i]  # do not confuse distutils parsing
        elif sys.argv[i].startswith("--profile="):
            if sys.argv[i][10:] == "yes": options.profile = True
            elif sys.argv[i][10:] == "no": options.profile = False
            else:
                print("Error: Argument to --profile must be yes or no")
                exit(1)
            del sys.argv[i]
        else:
            i += 1

    if options.use_cython not in (True, False, None):
        options.use_cython = None
    if options.use_cython is False:
        options.profile = False  # profiling has no sense in python build
    if options.profile not in (True, False):
        options.profile = False
    return options

def configure_build(options):
    """Configure build according to options previously returned by parse_cmdline_options()"""
    if options.use_cython is not False:
        # autodetect (or check for) cython
        try:
            from Cython.Distutils import build_ext
            from Cython.Distutils.extension import Extension
        except ImportError:
            if options.use_cython is True:
                print("Error: Cython was not found and --use-cython=yes was passed.")
                print("       please install cython in order to build faster PyBayes.")
                exit(1)
            else:  # use_cython is None (autodetect)
                print("Warning: Cython was not found on your system. Falling back to pure")
                print("         python mode which may be significantly slower.")
            options.use_cython = False
        else:
            if options.use_cython is not True:
                print("Notice: Cython found. Great!")
            options.build_ext = build_ext
            options.Extension = Extension

    if options.use_cython is not False:
        # determine path to NumPy C header files
        try:
            import numpy
        except ImportError:
            if options.use_cython is True:
                print("Error: Cannot import NumPy. It is needed at build-time in order to determine")
                print("       include path for it. NumPy is needed runtime for every PyBayes build")
                print("       and buid-time for cython build.")
                exit(1)
            else:
                print("Warning: Cython was found on your system, but NumPy was not. Numpy is needed")
                print("         build-time for cython builds and runtime for all builds. Falling back")
                print("         to pure python build.")
        else:
            if options.use_cython is not True:
                print("Notice: NumPy found. Good!")
            options.use_cython = True
            options.numpy_include_dir = numpy.get_include()
            del numpy

def main():
    options = parse_cmdline_options()
    configure_build(options)

    if options.use_cython is True:
        params['cmdclass'] = {'build_ext': options.build_ext}
        params['py_modules'] = ['pybayes.__init__', 'pybayes.stresses.__init__',
                                'pybayes.tests.__init__', 'pybayes.wrappers.__init__']
        params['ext_modules'] = []

        pxd_deps = ['__init__.pxd',
                    'filters.pxd',
                    'pdfs.pxd',
                    'stresses/stress_filters.pxd',
                    'wrappers/_linalg.pxd',
                    'wrappers/_numpy.pxd',
                    ]
        deps = ['pybayes/' + pxd_file for pxd_file in pxd_deps]  # dependency list
        deps.append('tokyo/tokyo.pxd')  # plus tokyo's pxd file
        # TODO: add cython's numpy.pxd as a dependency
        extensions = ['filters.py',
                    'pdfs.py',
                    'stresses/stress_filters.py',
                    'tests/support.py',
                    'tests/test_filters.py',
                    'tests/test_wrappers_numpy.py',
                    'tests/test_wrappers_linalg.py',
                    'tests/test_pdfs.py',
                    'wrappers/_linalg.pyx',
                    'wrappers/_numpy.pyx',
                    ]
        ext_options = {}  # options common to all extensions
        ext_options['include_dirs'] = [options.numpy_include_dir]
        ext_options['extra_compile_args'] = ["-O2"]
        ext_options['extra_link_args'] = ["-Wl,-O1"]
        ext_options['pyrex_c_in_temp'] = True  # do not pollute source directory with .c files
        ext_options['pyrex_directives'] = {'profile':options.profile, 'infer_types':True}
        ext_options['pyrex_include_dirs'] = ["tokyo"]  # find tokyo.pxd from bundled tokyo
        for extension in extensions:
            module = "pybayes." + os.path.splitext(extension)[0].replace("/", ".")
            paths = ["pybayes/" + extension]
            paths += deps  # simple "every module depends on all pxd files" logic
            params['ext_modules'].append(options.Extension(module, paths, **ext_options))

        # build and install bundled tokyo
        params['ext_modules'].append(options.Extension(
            'tokyo',  # module name
            ['tokyo/tokyo.pyx', 'tokyo/tokyo.pxd'],  # source file and deps
            libraries=['cblas', 'lapack'],
            **ext_options
        ))

    else:  # options.use_cython is False
        params['packages'] = ['pybayes', 'pybayes.stresses', 'pybayes.tests', 'pybayes.wrappers']

    setup(**params)

if __name__ == '__main__':
    main()
