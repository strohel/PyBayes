#!/usr/bin/env python
# -*- coding: utf-8 -*-

import integrate_cython as c
import integrate_python as p
import time

# Edit parameters here:

params = 0.0, 3.0, 1*10**8 # from, to, N (number of steps)

# Comment/Uncomment individual tests here>

tests = [
	("cython_typed_openmp", c.integrate_omp),
	("cython_typed", c.integrate_typed),
#	("cython", c.integrate),
#	("python", p.integrate),
]


# Do not edit below here

times = {}

print "Numerical integration from {0} to {1} of x^2 with {2} steps:".format(*params)
print

for (name, func) in tests:
	start_time = time.time()
	start_clock = time.clock()
	result = func(*params)
	times[name] = (time.time() - start_time, time.clock() - start_clock)
	print "{0:>19}: result = {1}; real time = {2}s; cpu time = {3}s".format(name, result, times[name][0], times[name][1])

print
print "Relative speedups (2-core Intel Core 2 Duo processor):"

for i in range(len(tests) - 1, 0, -1):
	print
	iname = tests[i][0]
	for j in range(0, i):
		jname = tests[j][0]
		print jname + "/" + iname + ":", times[iname][0]/times[jname][0]
