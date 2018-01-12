// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2018, the National Center for Supercomputing Applications
//
// Primary authors:
//     Valentin Pavlov <vpavlov@rila.bg>
//     Peicho Petkov <peicho@phys.uni-sofia.bg>
//     Stoyan Markov <markov@acad.bg>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include <stdexcept>
#include "factorizer.h"
#include "memory.h"

int ORG_NCSA_IRIS::factorize(int n, int **factors, int **powers)
{
    if(n <= 0) {
	throw std::invalid_argument("Can only factorize positive integers!");
    }

    *factors = new int[n/2];
    *powers = new int[n/2];
    int count = 0;
    for(int p = 2; n != 1; p++) {
	while(n % p == 0) {
	    if(count == 0 || (*factors)[count-1] != p) {
		(*factors)[count] = p;
		(*powers)[count++] = 1;
	    }else {
		(*powers)[count-1]++;
	    }
	    n /= p;
	}
    }

    return count;
}

static void next_fcomb(int fcnt, int *factors, int *powers,
			 int tx, int ty, int tz,
			 int **output, int *out_count)
{
    if(fcnt == 0) {
	output[*out_count][0] = tx;
	output[*out_count][1] = ty;
	output[*out_count][2] = tz;
	(*out_count)++;
	return;
    }

    for(int x = powers[0]; x>= 0; x--) {
	for(int i = 0; i < x; i++) {
	    tx *= factors[0];
	}

	for(int y = powers[0] - x; y >= 0; y--) {
	    for(int i = 0; i < y; i++) {
		ty *= factors[0];
	    }

	    for(int i = 0; i < powers[0]-x-y; i++) {
		tz *= factors[0];
	    }

	    next_fcomb(fcnt-1, factors+1, powers+1, tx, ty, tz,
		       output, out_count);

	    for(int i = 0; i < powers[0]-x-y; i++) {
		tz /= factors[0];
	    }

	    for(int i = 0; i < y; i++) {
		ty /= factors[0];
	    }
	}
        for (int i = 0; i < x; i++)
        {
	    tx /= factors[0];
        }
    }
}

int **ORG_NCSA_IRIS::grid_factorizations(int n, int *count)
{
    int *factors;
    int *powers;
    int fcnt = factorize(n, &factors, &powers);

    int prod = 1;
    for(int i = 0; i < fcnt; i++) {
	prod *= (powers[i]+2)*(powers[i]+1)/2;
    }

    int **retval;
    memory::create_2d(retval, prod, 3);

    *count = 0;
    next_fcomb(fcnt, factors, powers, 1, 1, 1, retval, count);

    delete [] factors;
    delete [] powers;
    return retval;
}

// OAOO: used in filter_factors_xxx procedures
#define REMOVE_FACTOR					\
    for(int j = 0; j < 3; j++) {			\
        factors[i][j] = factors[n-1][j];		\
    }							\
    n--;						\
    i--;

int ORG_NCSA_IRIS::filter_factors_exact(int n, int **factors, int *filter)
{
    for(int i=0;i<n;i++) {
	if((filter[0] != 0 && factors[i][0] != filter[0]) ||
	   (filter[1] != 0 && factors[i][1] != filter[1]) ||
	   (filter[2] != 0 && factors[i][2] != filter[2]))
	{
	    REMOVE_FACTOR
	}
    }
    return n;
}

int ORG_NCSA_IRIS::filter_factors_mod(int n, int **factors, int *filter)
{
    for(int i=0;i<n;i++) {
	if((filter[0] % factors[i][0]) ||
	   (filter[1] % factors[i][1]) ||
	   (filter[2] % factors[i][2]))
	{
	    REMOVE_FACTOR
	}
    }
    return n;
}

#undef REMOVE_FACTOR
