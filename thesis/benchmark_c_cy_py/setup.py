#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# extra_compile_args=["-O2", "-march=core2", "-fopenmp"

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            "integrate_cython",
            ["integrate_cython.pyx"],
            libraries=["integrate_c"],
            extra_compile_args=["-O2", "-fopenmp"],
            extra_link_args=["-Wl,-O1", "-fopenmp", "-L./"]
        )
    ]
)
