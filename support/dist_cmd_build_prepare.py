# Copyright (c) 2011 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
An additional subcommand to distutils' build to handle Python/Cython build of PyBayes
"""

from distutils.cmd import Command
from distutils.errors import DistutilsSetupError
import distutils.log as log
from distutils.sysconfig import get_config_var
from glob import glob
import os
import string


class build_prepare(Command):
    """Additional build step that is used to add Cython Extension per each module that is specified"""

    description = 'scan defined packages for python modules and inject Cython module per each'

    def initialize_options(self):
        self.ext_options = {}  # options common to all extensions
        self.ext_options['include_dirs'] = [self.distribution.numpy_include_dir]
        self.ext_options['extra_compile_args'] = ['-O2']
        #self.ext_options['extra_link_args'] = ['-Wl,-O1']
        self.ext_options['pyrex_c_in_temp'] = True  # do not pollute source directory with .c files
        self.ext_options['pyrex_directives'] = {
            'profile':self.distribution.profile,
            'infer_types':True,
            "binding": False,  # default was changed to True in Cython commit 621dbe6403 and it
                               # breaks the build. I don't know what it means, it's undocumented.
        }
        self.deps = []  # .pxd dependencies for injected packages

    def finalize_options(self):
        # these options are passed through global distribution
        dist = self.distribution

        # these are just aliases to distribution variables
        self.packages = self.distribution.packages
        self.py_modules = self.distribution.py_modules
        self.package_data = self.distribution.package_data
        self.package_dir = {}
        if self.distribution.package_dir:
            for name, path in self.distribution.package_dir.items():
                self.package_dir[name] = convert_path(path)

        if self.py_modules:
            raise DistutilsSetupError("PyBayes-tweaked distutils doesn't support nonempty `py_modules`")
        if not self.packages:
            raise DistutilsSetupError("PyBayes-tweaked distutils doesn't support nempty `packages`")

    def run(self):
        build_py = self.distribution.get_command_obj('build_py')
        self.get_package_dir = build_py.get_package_dir  # borrow a method from build_py
        self.check_package = build_py.check_package  # ditto

        for package in self.packages:
            package_dir = self.get_package_dir(package)
            self.inject_package_modules(package, package_dir)

        self.update_dependencies()

    def inject_package_modules (self, package, package_dir):
        """This is our purpose and a hack - we create Cython extensions here"""
        self.check_package(package, package_dir)
        py_files = glob(os.path.join(package_dir, "*.py"))
        pyx_files = glob(os.path.join(package_dir, "*.pyx"))
        for pyx_file in pyx_files:  # subtract py files that also have a pyx file
            corres_py_file = pyx_file[:-3]+"py"  # corresponding .py file
            if corres_py_file in py_files:
                del py_files[py_files.index(corres_py_file)]
        pxd_files = glob(os.path.join(package_dir, "*.pxd"))
        self.deps += pxd_files
        setup_script = os.path.abspath(self.distribution.script_name)

        if package not in self.package_data:
            self.package_data[package] = []
        # per-package data have to have path relative to their package
        self.package_data[package] += [os.path.basename(pxd_file) for pxd_file in pxd_files]

        for f in py_files + pyx_files:
            basename = os.path.basename(f)
            if os.path.abspath(f) == setup_script:
                self.debug_print("excluding %s" % setup_script)
                continue
            if basename == '__init__.py':
                # otherwise import package (`import pybayes`) does not really work
                continue
            if basename == '__main__.py':
                # otherwise `python -m pybayes.tests` does not really work
                continue

            module = os.path.splitext(f)[0].replace("/", ".")
            self.inject_extension(module, f)

    def inject_extension(self, module, f):
        log.info("injecting Cython extension {0} (module {1})".format(f, module))
        self.distribution.ext_modules.append(
            self.distribution.Extension(module, [f], **self.ext_options)
        )

    def update_dependencies(self):
        """Update dependencies of all already injected extensions"""
        for module in self.distribution.ext_modules:
            module.sources += self.deps
