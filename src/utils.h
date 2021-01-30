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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_UTILS_H__
#define __IRIS_UTILS_H__

#include <stdlib.h>
#include "real.h"

namespace ORG_NCSA_IRIS {
    
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

#define ROW_MAJOR_OFFSET(x, y, z, ny, nz) ((z) + (nz)*((y) + (ny)*(x)))

    typedef iris_real (*simple_fn)(iris_real x, void *obj);

    iris_real *gauss(int n, iris_real **a, iris_real *b);
    void pade_approximant(int m, int n, iris_real *input,
			  iris_real *&out_nom,
			  iris_real *&out_denom);

    int binom(int n, int k);

    iris_real root_of(simple_fn fn, iris_real x0, void *obj);  // using secant method

    void qsort_int(int *in_data, size_t in_num);

    inline bool is_power_of_2(int x) { return (x & (x - 1)) == 0; };

    inline int rand_int(int from, int to) { return (rand() % (to-from+1)) + from; };
    
    template <typename T>
    void shuffle(T *a, int n) {
	T tmp;
	for(int i=n-1;i>0;i--) {
	    int j = rand_int(0, i);
	    tmp = a[j];
	    a[j] = a[i];
	    a[i] = tmp;
	}
    }

    int get_int_env(const char *name, int def);

    float get_float_env(const char *name, float def);
}

#endif
