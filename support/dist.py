# Copyright (c) 2011 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
An extension to distutils' Distribution to handle Python/Cython build of PyBayes
"""

from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
import distutils.log as log
from distutils.util import strtobool

from dist_cmd_build import PyBayesBuild
from dist_cmd_build_prepare import PyBayesBuildPrepare
from dist_cmd_test import PyBayesTest


class PyBayesDistribution(Distribution):
    """An extension to distutils' Distribution that provides Cython/Python build switching etc."""

    def __init__(self, attrs=None):
        Distribution.__init__(self, attrs)
        self.use_cython = None
        self.profile = False
        if not self.ext_modules:
            self.ext_modules = []
        self.cmdclass['test'] = PyBayesTest

        self.global_options += [
            ('use-cython=', None, "use Cython to make faster binary python modules (choices: "
             + "yes/no; default: autodetect)"),
            ('profile=', None, 'embed profiling information into Cython build (choices: '
             + 'yes/no; default: no)')
        ]

    def has_ext_modules(self):
        if self.use_cython:
            return True
        return Distribution.has_ext_modules(self)

    def parse_command_line(self):
        """We need to process the options once command line is parsed"""
        ret = Distribution.parse_command_line(self)
        if ret:
            self.finalize_command_line()
        return ret

    def finalize_command_line(self):
        if self.profile not in (True, False):
            self.profile = bool(strtobool(self.profile))
        if self.use_cython is None:
            self.use_cython = self._find_cython()
            if self.use_cython:
                log.info("Cython and NumPy found, enabling optimised Cython build.")
            else:
                log.info("Cython or NumPy was not found on your system. Falling back to pure")
                log.info("python mode which may be somehow slower.")
        elif self.use_cython not in (True, False):
            requested = strtobool(self.use_cython)
            if requested and not self._find_cython():
                raise DistutilsOptionError("Cython build was requested but no or too old Cython"
                                            + " found on your system.")
            self.use_cython = bool(requested)
            if self.use_cython:
                log.debug("Cython build requested and Cython and NumPy found.")
            else:
                log.debug("Pure Python build requested, not searching for Cython.")
        if self.use_cython:
            self.finalize_cython_options()

    def finalize_cython_options(self):
        self.cmdclass['build'] = PyBayesBuild
        self.cmdclass['build_prepare'] = PyBayesBuildPrepare
        self.cmdclass['build_ext'] = self.build_ext

        # .pyc files just litter site-packages in Cython build
        install = self.get_command_obj('install')
        install.compile = 0
        install.optimise = 0

    def _find_cython(self):
        """Returns true if sufficient version of Cython in found on system, false otherwise.

        If true is returned, also sets some variables useful for Cython build
        """
        # autodetect (or check for) cython
        try:
            from Cython.Distutils import build_ext
            from Cython.Distutils.extension import Extension
        except ImportError:
            return False
        self.build_ext = build_ext
        self.Extension = Extension

        # determine path to NumPy C header files
        try:
            import numpy
        except ImportError:
            log.warn("Cython was found on your system, but NumPy was not. NumPy is needed")
            log.warn("build-time for cython builds and runtime for all builds. Falling back")
            log.warn("to pure Python build.")
            return False
        self.numpy_include_dir = numpy.get_include()
        return True
