# Copyright (c) 2011 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""
An extension distutils' build to handle Python/Cython build of PyBayes
"""

from distutils.command.build import build


class PyBayesBuild(build):
    """Introduce additional build step to inject Cython extensions"""

    def finalize_options(self):
        build.finalize_options(self)
        # prepend our custom command
        self.sub_commands = [('build_prepare', None)] + self.sub_commands
