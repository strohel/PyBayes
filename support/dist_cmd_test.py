# Copyright (c) 2011 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
A custom command for distutils to facilitate testing of PyBayes
"""

from distutils.cmd import Command
from distutils.errors import DistutilsExecError
import distutils.log as log
from os.path import abspath, dirname, join
import sys
import unittest


class test(Command):
    """Test PyBayes in the build directory"""

    description = 'run unit test-suite of PyBayes within build directory'
    user_options = [
        ('fatal', None,
         'make test failures fatal [default]'),
        ('non-fatal', None,
         'make test failures fatal (opposite of `--fatal`)')
    ]
    boolean_options = ['fatal']
    negative_opt = {'non-fatal':'fatal'}

    def initialize_options(self):
        self.build_lib = None
        self.fatal = 1

    def finalize_options(self):
        self.set_undefined_options('build', ('build_lib', 'build_lib'))

    def run(self):
        self.run_command('build')  # build if not alredy run
        orig_path = sys.path[:]
        try:
            tests_path = abspath(self.build_lib)
            sys.path.insert(0, tests_path)
            import pybayes.tests
            if dirname(pybayes.tests.__file__) != join(tests_path, 'pybayes', 'tests'):
                raise Exception("Expected that imported pybayes.tests would be from "
                                + "{0}, but it was from {1} instead".format(tests_path,
                                dirname(pybayes.tests.__file__)))
            suite = unittest.TestLoader().loadTestsFromModule(pybayes.tests)
            result = unittest.TextTestRunner(verbosity=self.verbose).run(suite)
            if not result.wasSuccessful():
                raise Exception("There were test failures")
        except Exception as e:
            if self.fatal:
                raise DistutilsExecError(e)
            else:
                log.warn("ignoring exception: {0}".format(e))
        finally:
            sys.path = orig_path
