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
#include <string.h>
#include <stdexcept>
#include <cmath>
#include "utils.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

#define H 1.42e-7
#define SMALL 1e-15

// Calculate the binomial coefficient (n|k) using the multiplicative formula
int ORG_NCSA_IRIS::binom (int n, int k)
{
    iris_real retval = 1;
    for(int i=1;i<=k;i++) {
	retval *= (n+1-i)/(iris_real)i;
    }
    return (int)retval;
}

// Gaussian elimination. Nothing fancy, no pivoting, etc. Should be enough for
// the type of systems we're solving here.
iris_real *ORG_NCSA_IRIS::gauss(int n, iris_real **a, iris_real *b)
{
    iris_real *x;
    memory::create_1d(x, n, true);
    
    // Gaussian elimination

    for(int k=0;k<n-1;k++) {
	for(int i=1+k;i<n;i++) {
	    a[i][k] /= a[k][k];
	    for(int j=1+k;j<n;j++) {
		a[i][j] -= a[i][k] * a[k][j];
	    }
	}
    }

    // Forward elimination
    for(int k=0;k<n-1;k++) {
	for(int i=1+k;i<n;i++) {
	    b[i] -= a[i][k] * b[k];
	}
    }

    // Backward solve
    for(int i=n-1;i>=0;i--) {
	iris_real s = b[i];
	for(int j=i+1;j<n;j++) {
	    s -= a[i][j] * x[j];
	}
	x[i] = s / a[i][i];
    }

    return x;
}


// Compute the Pade approximant P[m, n] from the coefficients of the Taylor
// expansion (given in the static taylor_coeff above).
//
// We start with the Tyalor expansion, given through the coefficients C:
//
// c0 + c1*x + c2*x^2 +... = (a0 + a1*x + a2*x^2 +...) / (1 + b1*x + b2*x^2+...)
//
// M is the number of a's and N is the number of b's required.
//
// Multiplying through the denominator we get a system of equations
//
// a0 = c0
// a1 = c1 + c0 * b1
// a2 = c2 + c1 * b1 + c0 * b2
// ...
//
// We cut off the system to M+N+1 equations of M+N+1 unknowns (a's and b's).
//
// A complete example:
//
// Start with a taylor expansion of d2/dx2 up to second order:
//
//  (d2/dx2-taylor-coeff 2)
//
// gives back 
//
//  #(1 -1/12)
//
// This means that d2/dx2 =~ δ^2/h^2 * (1 - 1/12 δ^2). We deal only with the 
// expression in the brackets, which is
//
// 1*δ^0 + 0*δ^1 - (1/12)*δ^2,
//
// so C is (list 1 0 -1/12)
//
// Since the highest degree is 2, M+N must = 2. The case of N = 0 is trivial,
// since it is just equal to the Taylor expansion. Let's try M = 0, N = 2.
// For this case we have
//
// 1*δ^0 + 0*δ^1 - (1/12)*δ^2 = (a0 + a1*δ + a2*δ^2) / (1 + b1*δ + b2*δ^2)
//
// a0 = c0
//  0 = c1 + c0*b1
//  0 = c2 + c1*b1 + c0*b2
//
// or in matrix form
//
// |1  0  0| * |a0| = | c0|
// |0 c0  0|   |b1|   |-c1|
// |0 c1 c0|   |b2|   |-c2|
//
// |1 0 0| * |a0| = |   1|
// |0 1 0|   |b1| = |   0|
// |0 0 1|   |b2| = |1/12|
//
// So, according to the Pade approximation P[0,2],
//
// d2/dx2 =~ δ^2/h^2 * 1 / (1 + 1/12 δ^2)

// This function returns the square matrix for the left hand side.
//
// The matrix is M+N+1 square and is constructed as follows:
//
// 1. Starting from the top-left, we put (M+1)x(M+1) identity matrix
// 2. Directly below, put Nx(M+1) zero matrix. With this, we have a M+N+1 rows
//    of M+1 columns. What's left is to add to the right N columns.
// 3. The i-th such column (starting from 1) starts with i zeros and continues
//    with c0, c1, c2, etc. up till there are M+N+1 elements in the column.
//    If the corresponding row is in the first M+1 rows, add minus sign.
//
// E.g. for a [2, 2] we need to make 2+2+1 = 5x5 square matrix A:
//
// A = |. . . . .|, then after step 1: A = |1 0 0 . .|, then after step 2:
//     |. . . . .|                         |0 1 0 . .|
//     |. . . . .|                         |0 0 1 . .|
//     |. . . . .|                         |. . . . .|
//     |. . . . .|                         |. . . . .|
//
// A = |1 0 0 . .|, then, after Step 3 first column: A = |1 0 0   0 .|
//     |0 1 0 . .|                                       |0 1 0 -c0 .|
//     |0 0 1 . .|                                       |0 0 1 -c1 .|
//     |0 0 0 . .|                                       |0 0 0  c2 .|
//     |0 0 0 . .|                                       |0 0 0  c3 .|
//
// and finally A = |1 0 0   0   0|
// 		   |0 1 0 -c0   0|
// 		   |0 0 1 -c1 -c0|
// 		   |0 0 0  c2  c1|
// 		   |0 0 0  c3  c2|"
static iris_real **compute_a(int m, int n, iris_real *input)
{
    iris_real **a;
    memory::create_2d(a, m+n+1, m+n+1, true);

    // Step 1
    for(int i=0;i<=m;i++) {
	a[i][i] = 1.0;
    }

    // Step 2 is taken care from the initialization above

    // Step 3
    for(int i=m+1; i <= m+n; i++) {
	for(int j=i-m; j <= m+n; j++) {
	    iris_real c = input[j - i + m];
	    if(j <= m) {
		c *= -1.0;
	    }
	    a[j][i] = c;
	}
    }
    return a;
}

// Right hand side of the same is build as follows (see above comments):
//
// Get all the c's one after the other, the first m+1 of them being with a
// positive sign, the rest with negative
static iris_real *compute_b(int m, int n, iris_real *input)
{
    iris_real *b;
    memory::create_1d(b, m+n+1);
    for(int i=0;i<=m+n;i++) {
	iris_real c = input[i];
	if(i > m) {
	    c *= -1.0;
	}
	b[i] = c;
    }
    return b;
}

void ORG_NCSA_IRIS::pade_approximant(int m, int n,
				     iris_real *input,
				     iris_real *&out_nom,
				     iris_real *&out_denom)
{
    iris_real **a = compute_a(m, n, input);
    iris_real *b = compute_b(m, n, input);
    iris_real *pc = gauss(m+n+1, a, b);

    memory::create_1d(out_nom, m+1);
    memory::create_1d(out_denom, n+1);
    out_denom[0] = 1.0;
    memcpy(out_nom, pc, (m+1)*sizeof(iris_real));
    if(n != 0) {
	memcpy(out_denom+1, pc+m+1, n*sizeof(iris_real));
    }

    memory::destroy_2d(a);
    memory::destroy_1d(b);
    memory::destroy_1d(pc);
}

// Find the root of the simple function FN using a finite-difference
// approximation of Newton-Raphson's method (you can also say this is
// the secant method)
iris_real ORG_NCSA_IRIS::root_of(simple_fn fn, iris_real x0, void *obj)
{
    iris_real x = x0;
    for(int i=0;i<10;i++) {
	iris_real fx = fn(x, obj);
	if(fabs_fn(fx) < SMALL) {
	    break;
	}
	x -= fx*H/(fn(x+H, obj) - fx);
    }
    return x;
}

