#include "integrate_c.h"  /* just to assure that declarations match */

#include <stdio.h>
#include <stdlib.h>

double f(double x)
{
	return x*x;
}

double lib_integrate_c(double a, double b, int N)
{
	double s = 0.0;
	double dx = (b-a)/N;
	int i;

	if(dx == 0.0) {
		fprintf(stderr, "dx == 0!\n");
		return 0.0;
	}

	for(i = 0; i < N; i++) {
		s += f(a + (i + 1./2.)*dx)*dx;
	}
	return s;
}

double lib_integrate_c_omp(double a, double b, int N)
{
	double s = 0.0;
	double dx = (b-a)/N;
	int i;


	if(dx == 0.0) {
		fprintf(stderr, "dx == 0!\n");
		return 0.0;
	}

	#pragma omp parallel for reduction(+:s) lastprivate(a)
	for(i = 0; i < N; i++) {
		s += f(a + (i + 1./2.)*dx)*dx;
	}
	return s;
}
