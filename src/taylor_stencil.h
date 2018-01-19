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
#ifndef __IRIS_TAYLOR_STENCIL_H__
#define __IRIS_TAYLOR_STENCIL_H__

#include "stencil.h"

namespace ORG_NCSA_IRIS {

    // A Laplacian stencil derived from the Taylor expansion of the
    // second derivative approximation with a central difference operator.
    //
    // From the theory of difference operators, we have:
    // 
    // d2/dx^2 =~ 4/hx^2 * asinh(δx/2)^2
    //
    // Which expanded in Taylor series:
    // 
    // d2/dx^2 =~ 1/hx^2 * δx^2 { 1 - 1/12 δx^2 + 1/90 δx^4 - 1/560 δx^6 ... }
    //
    // A second-order accurate scheme is derived by considering the first term
    // only: δx^2/hx^2. This is the most basic central difference approximation.
    //
    // A fourth-order scheme is derived by considering the first two terms;
    // A sixth-order scheme is derived by considering the first three terms;
    // Etc.

    class taylor_stencil : public stencil {

    public:
	taylor_stencil(class iris *in_obj, int in_order);
	~taylor_stencil();

	void commit();

    public:
	iris_real m_h2;  // mesh's hx squared
	int m_order;     // Accuracy order, e.g. 2nd, 4th, 6th, etc.
    };
}

#endif
