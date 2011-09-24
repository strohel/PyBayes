#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import integrate_cython as c
except ImportError as e:
    print "Failed to import integrate_cython, cython tests wont be available:", e
    c = None
import integrate_python as p
import time

# Edit parameters here:

params = 0.0, 3.0, 200*10**6  # from, to, N (number of steps)

# Comment/Uncomment individual tests here>

tests = []
if c:
    tests += [("c_omp", c.integrate_c_omp),
              ("cy_typed_omp", c.integrate_omp),
              ("c", c.integrate_c),
              ("cython_typed", c.integrate_typed),
              ("cython", c.integrate)]
tests += [("python", p.integrate)]


# No need to edit edit below here

times = {}

print "Numerical integration from {0} to {1} of x^2 with {2} steps:".format(*params)
print

for (name, func) in tests:
    start_time = time.time()
    start_clock = time.clock()
    result = func(*params)
    times[name] = (time.time() - start_time, time.clock() - start_clock)
    print "{0:>12}: result = {1}; real time = {2}s; cpu time = {3}s".format(name, result, times[name][0], times[name][1])

print
print "Relative speedups:"

for i in range(len(tests) - 1, 0, -1):
    print
    iname = tests[i][0]
    for j in range(i-1, -1, -1):
        jname = tests[j][0]
        print jname + "/" + iname + ":", times[iname][0]/times[jname][0]
