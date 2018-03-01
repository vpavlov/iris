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
#ifndef __IRIS_CDO3D_H__
#define __IRIS_CDO3D_H__

#include "real.h"

namespace ORG_NCSA_IRIS {

    // A 3D central difference operator stencil
    class cdo3D {

    public:
	// Create a 3D central difference operator stencil of accuracy order N
	// (must be even) with a constant coefficient C, and X, Y and Z powers
	// of δ being XP, YP and ZP respectively.
	// Thus,
	// 
	// new cdo(2, 144, 0, 0, 0) will return a 27-point stencil with a
	// central element 144
	//
	// new cdo(2, 144, 2, 0, 0) will return a 27-point stencil for δ^2x
	// new cdo(2, 144, 2, 2, 2) for δ^2x + δ^2y + δ^2z
	
	cdo3D(int n, iris_real c, int xp, int yp, int zp);
	~cdo3D();

	void dump();

	void operator += (cdo3D &other);

    private:

	// Return the central difference operator of order n (δ^n) coefficients 
	//
	// The coefficients are n+1 and follow the formula:
	// kth coeff is (-1)^k (n k)
	//
	//For example:
	// for n = 1 coefficients are { 1, -1 }
	// for n = 2 coefficients are { 1, -2, 1 }
	// for n = 3 coefficients are { 1, -3, 3, -1 }
	// for n = 4 coefficients are { 1, -4, 6, -4, 1 }, etc.
	int *coeff(int n);

	int m_n;  // order
	iris_real ***m_data;
    };
}

#endif
