# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

cpdef double f(double x):
	return x*x

cpdef integrate(a, b, N):
	s = 0
	dx = (b-a)/N

	if dx == 0:
		print "dx == 0!"
		return 0

	for i in xrange(N):
		a = (i+1/2)*dx
		s += f(a)*dx
	return s

cpdef double integrate_typed(double a, double b, int N) except 0:
	cdef double s = 0
	cdef double dx = (b-a)/N
	cdef int i

	if dx == 0:
		print "dx == 0!"
		return 0

	for i in xrange(N):
		a = (i+1/2)*dx
		s += f(a)*dx
	return s

cpdef double integrate_omp(double a, double b, int N) except 0:
	cdef double s = 0
	cdef double dx = (b-a)/N
	cdef int i

	if dx == 0:
		print "dx == 0!"
		return 0

	for i in xrange(N):
		a = (i+1/2)*dx
		s += f(a)*dx
	return s
