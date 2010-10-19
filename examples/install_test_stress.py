#!/usr/bin/env python
# Copyright (c) 2010 Matej Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Install PyBayes and run tests and stresses"""

from optparse import OptionParser
from os.path import abspath, dirname, exists, join
from string import join as str_join
from subprocess import call, check_call


def parse_options():
    def_pybayes_dir = abspath(dirname(dirname(__file__)))
    def_data_dir = join(def_pybayes_dir, 'examples', 'stress_data')

    parser = OptionParser(description='Install, test and stress possible multiple ' +
                          'variants of PyBayes in one go')
    parser.add_option('-b', '--pybayes-dir', dest='pybayes_dir', action='store', default=def_pybayes_dir,
                      help='directory from where to install PyBayes; current: %default')
    parser.add_option('-m', '--mode', dest='modes', action='append', type='choice',
                      choices=('p', 'c', 'a'),
                      help='which mode to build & test PyBayes in; may be specified multiple times; ' +
                      'valid modes are: p[ython], c[ython], a[uto]; default: a')
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
        options.modes = ['auto']
    if args:
        print "Error: unparsed arguments left on command line"
        parser.print_help()
        exit(1)
    return options

def install(mode, options):
    modes = {'p':['--use-cython=no'],
             'c':['--use-cython=yes'],
             'a':[]}
    profiles = {True:['--profile=yes'],
                False:['--profile=no'],
                None:[]}

    setup_py = join(options.pybayes_dir, 'setup.py')
    if not exists(setup_py):
        raise RuntimeError('{0} does not exist!'.format(setup_py))

    args = [setup_py]
    args.extend(modes[mode])
    args.extend(profiles[options.profile])

    commands = []
    if options.force_rebuild:
        commands.append('clean')
    commands.append('install')

    for command in commands:
        cmdargs = args[:]
        cmdargs.append(command)
        print(str_join(cmdargs, ' '))
        check_call(cmdargs)

def run_tests(options):
    run_tests = join(options.pybayes_dir, 'examples', 'run_tests.py')
    args = [run_tests]
    print(str_join(args, ' '))
    call(args)

def run_stresses(options):
    run_tests = join(options.pybayes_dir, 'examples', 'run_stresses.py')
    args = [run_tests, '-d', options.data_dir]
    print(str_join(args, ' '))
    call(args)

def main():
    options = parse_options()

    for mode in options.modes:
        install(mode, options)

        if options.run_tests:
            run_tests(options)

        if options.run_stresses:
            run_stresses(options)

if __name__ == '__main__':
    main()
