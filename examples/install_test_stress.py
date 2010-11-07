#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Install PyBayes and run tests and stresses"""

from distutils.dist import Distribution
from distutils.sysconfig import get_python_lib
from optparse import OptionParser
import os
import shutil
from string import join
from subprocess import call, check_call


def parse_options():
    def_pybayes_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    def_data_dir = os.path.join(def_pybayes_dir, 'examples', 'stress_data')

    parser = OptionParser(description='Install, test and stress possible multiple ' +
                          'variants of PyBayes in one go')
    parser.add_option('-b', '--pybayes-dir', dest='pybayes_dir', action='store', default=def_pybayes_dir,
                      help='directory from where to install PyBayes; current: %default')
    parser.add_option('-m', '--mode', dest='modes', action='append', type='choice',
                      choices=('p', 'c', 'a'),
                      help='which mode to build & test PyBayes in; may be specified multiple times; ' +
                      'valid modes are: p[ython], c[ython], a[uto]; default: -m c -m a')
    parser.add_option('-c', '--clean', dest='clean', action='store_true',
                      default=False, help='clean installed PyBayes before build. May be DANGEROUS as it ' +
                      'deletes PyBayes directory and everything underneath it; if you mix cython & python ' +
                      'build, you should however enable this; default: do not remove anything')
    parser.add_option('-f', '--force-rebuild', dest='force_rebuild', action='store_true',
                      default=False, help='force recython & rebuild even if fresh built objects exist; ' +
                      'default: reuse built objects if up-to-date')
    parser.add_option('-p', '--profile', dest='profile', action='store_true',
                      default=None, help='embend profiling information into PyBayes cython build; ' +
                      'default: let PyBayes\' setup.py decide')
    parser.add_option('-T', '--no-tests', dest='run_tests', action='store_false',
                      default=True, help='do not run PyBayes tests upon build; default: run tests')
    parser.add_option('-S', '--no-stresses', dest='run_stresses', action='store_false',
                      default=True, help='do not run PyBayes stress suite upon build; default: run stresses')
    parser.add_option('-d', '--data-dir', dest='data_dir', action='store',
                      default=def_data_dir, help='directory constaining data for stresses; current: %default')
    (options, args) = parser.parse_args()
    if not options.modes:
        options.modes = ['p', 'a']  # test python & cython, but do not fail when cython is unavailable
    if args:
        print "Error: unparsed arguments left on command line"
        parser.print_help()
        exit(1)
    return options

def clean(options):
    orig_dir = os.getcwd()
    os.chdir(options.pybayes_dir)  # so that distutils can source setup.cfg if it exists

    dist = Distribution()
    dist.parse_config_files()
    prefix = dist.get_option_dict("install")["prefix"][1]  # get prefix out of parsed options
    install_dir = os.path.join(get_python_lib(False, False, prefix), "pybayes")  # we don't know if we should use plat_specific or no (depends on previous PyBayes install)
    del dist

    if os.path.isdir(install_dir):
        print "Recursively deleting {0}".format(install_dir)
        shutil.rmtree(install_dir)

    os.chdir(orig_dir)

def install(mode, options):
    modes = {'p':['--use-cython=no'],
             'c':['--use-cython=yes'],
             'a':[]}
    profiles = {True:['--profile=yes'],
                False:['--profile=no'],
                None:[]}

    if not os.path.isdir(options.pybayes_dir):
        raise RuntimeError('{0} does not exist!'.format(setup_py))

    args = ['./setup.py']
    args.extend(modes[mode])
    args.extend(profiles[options.profile])

    commands = []
    if options.force_rebuild:
        commands.append('clean')
    commands.append('install')

    orig_dir = os.getcwd()
    os.chdir(options.pybayes_dir)
    for command in commands:
        cmdargs = args[:]
        cmdargs.append(command)
        print(join(cmdargs, ' '))
        check_call(cmdargs)
    os.chdir(orig_dir)

def run_tests(options):
    run_tests = os.path.join(options.pybayes_dir, 'examples', 'run_tests.py')
    args = [run_tests]
    print(join(args, ' '))
    call(args)

def run_stresses(options):
    run_tests = os.path.join(options.pybayes_dir, 'examples', 'run_stresses.py')
    args = [run_tests, '-d', options.data_dir]
    print(join(args, ' '))
    call(args)

def main():
    options = parse_options()

    for mode in options.modes:
        if options.clean:
            clean(options)

        install(mode, options)

        if options.run_tests:
            run_tests(options)

        if options.run_stresses:
            run_stresses(options)

if __name__ == '__main__':
    main()
