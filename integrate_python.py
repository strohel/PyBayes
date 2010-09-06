# -*- coding: utf-8 -*-

def f(x):
	return x*x

def integrate(a, b, N):
	s = 0
	dx = (b-a)/N

	if dx == 0:
		print "dx == 0!"
		return 0

	for i in xrange(N):
		a = (i+1/2)*dx
		s += f(a)*dx
	return s
