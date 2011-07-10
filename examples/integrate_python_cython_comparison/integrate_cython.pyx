# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

from cython.parallel cimport prange

cdef extern from "integrate_c.h":
    double lib_integrate_c(double a, double b, int N)
    double lib_integrate_c_omp(double a, double b, int N)

cpdef double f(double x) nogil:
    return x*x

cpdef integrate(a, b, N):
    s = 0
    dx = (b-a)/N

    if dx == 0:
        print "dx == 0!"
        return 0

    for i in xrange(N):
        s += f(a + (i + 1./2.)*dx)*dx
    return s

cpdef double integrate_typed(double a, double b, int N):
    cdef double s = 0
    cdef double dx = (b-a)/N
    cdef int i

    if dx == 0:
        print "dx == 0!"
        return 0

    for i in xrange(N):
        s += f(a + (i + 1./2.)*dx)*dx
    return s

cpdef double integrate_omp(double a, double b, int N):
    cdef double s = 0
    cdef double dx = (b-a)/N
    cdef int i

    if dx == 0:
        print "dx == 0!"
        return 0

    for i in prange(N, nogil=True, schedule=guided):
        s += f(a + (i + 1./2.)*dx)*dx
    return s

cpdef double integrate_c(double a, double b, int N):
    return lib_integrate_c(a, b, N)

cpdef double integrate_c_omp(double a, double b, int N):
    return lib_integrate_c_omp(a, b, N)
