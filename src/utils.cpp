// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#include "utils.h"
#include "iris_exception.h"


namespace ORG_NCSA_IRIS {

// Brute force factorization of N into factors.
// fac is initialized here and will contain the unique prime factors of N
// mfac is initialized here and will contain the power of each primer factor
// returns the number of unique factors (sizeof fac)
//
// Of course, this can only be used for small Ns (e.g. number of ranks in a
// MPI communicator)

int factorize(int n, int **factors, int **powers)
{
    if(n <= 0) {
	throw new iris_exception("Can only factorize positive integers!");
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

}
