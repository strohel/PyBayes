#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.core import setup
import os.path

from support import determine_pybayes_version
from support.dist import PyBayesDistribution


dir = os.path.dirname(os.path.realpath(__file__))
version = determine_pybayes_version(dir=dir, fallback='0.3-post-nongit')

params = {
    # meta-data; see http://docs.python.org/distutils/setupscript.html#additional-meta-data
    'name':'PyBayes',
    'version':version,
    'author':'Matěj Laitl',
    'author_email':'matej@laitl.cz',
    'maintainer':'Matěj Laitl',
    'maintainer_email':'matej@laitl.cz',
    'url':'https://github.com/strohel/PyBayes',
    'description':'Python library for recursive Bayesian estimation (Bayesian filtering)',
    'long_description':'PyBayes is an object-oriented Python library for recursive Bayesian ' +
        'estimation (Bayesian filtering) that is convenient to use. Already implemented are ' +
        'Kalman filter, particle filter and marginalized particle filter, all built atop of ' +
        'a light framework of probability density functions. PyBayes can optionally use Cython ' +
        'for lage speed gains (Cython build can be several times faster in some situations).',
    # Note to myself: must manually upload on each release!
    'download_url':'https://github.com/downloads/strohel/PyBayes/PyBayes-'+version+'.tar.gz',
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
    ],

    # Python (Cython) packages to build:
    'packages':['pybayes', 'pybayes.stresses', 'pybayes.tests', 'pybayes.wrappers'],
    'package_data':{'pybayes.stresses':['data/stress_kalman_data.mat']},

    # THE trick - custom distribution that handles Python/Cython build
    'distclass':PyBayesDistribution
}

if __name__ == '__main__':
    setup(**params)
