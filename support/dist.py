# Copyright (c) 2011 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
An extension to distutils' Distribution to handle Python/Cython build of PyBayes
"""

from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils.util import strtobool

from dist_cmd_build import PyBayesBuild
from dist_cmd_build_prepare import PyBayesBuildPrepare


class PyBayesDistribution(Distribution):
    """An extension to distutils' Distribution that provides Cython/Python build switching etc."""

    def __init__(self, attrs=None):
        Distribution.__init__(self, attrs)
        self.ext_modules = []
        self.use_cython = None
        self.profile = False

        self.global_options += [
            ('use-cython=', None, "use Cython to make faster binary python modules (choices: "
             + "yes/no; default: autodetect)"),
            ('profile=', None, 'embed profiling information into Cython build (choices: '
             + 'yes/no; default: no)')
        ]

    def get_command_class(self, command):
        """Overriden method to return our custom command in case that Cython build is in effect"""
        self._ensure_options_finalised()
        if self.use_cython is not True:
            return Distribution.get_command_class(self, command)

        if command == "build":
            return PyBayesBuild
        if command == "build_prepare":
            return PyBayesBuildPrepare
        if command == "build_ext":
            return self.build_ext
        return Distribution.get_command_class(self, command)

    def has_ext_modules (self):
        """This method needs to be overriden as we create our Extension instances late in the
        cycle"""
        return True

    def _ensure_options_finalised(self):
        if self.profile not in (True, False):
            self.profile = bool(strtobool(self.profile))
        if self.use_cython is None:
            self.use_cython = self._find_cython()
            if self.use_cython:
                print("Notice: Cython and NumPy found, enabling optimised Cython build.")
            else:
                print("Notice: Cython was not found on your system. Falling back to pure")
                print("        python mode which may be somehow slower.")
        elif self.use_cython not in (True, False):
            requested = strtobool(self.use_cython)
            if requested and not self._find_cython():
                raise DistutilsOptionError("Cython build was requested but no or too old Cython"
                                            + " found on your system.")
            self.use_cython = bool(requested)

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
            print("Notice: Cython was found on your system, but NumPy was not. NumPy is needed")
            print("        build-time for cython builds and runtime for all builds. Falling back")
            print("        to pure Python build.")
            return False
        self.numpy_include_dir = numpy.get_include()
        return True
