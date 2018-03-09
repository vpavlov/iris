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
#ifndef __IRIS_LAPLACIAN3D_H__
#define __IRIS_LAPLACIAN3D_H__

#include "stencil.h"
#include "real.h"

namespace ORG_NCSA_IRIS {

    //
    // A symmetrical stencil that approximates a 3D Laplacian:
    // d2/dx^2 + d2/dy^2 + d2/dz^2
    //
    class laplacian3D : public stencil {
	
    public:
	laplacian3D(int in_acc) : stencil(3, 2, in_acc) {};
	~laplacian3D() {};

	void set_hx(iris_real in_hx) { m_hx = in_hx; m_dirty = true; }
	void set_hy(iris_real in_hy) { m_hy = in_hy; m_dirty = true; }
	void set_hz(iris_real in_hz) { m_hz = in_hz; m_dirty = true; }

	virtual void commit() = 0;

	// How many additional layers beyond the center we look at?
	// E.g. for a 7-point stencil or similar, we look at center + 1 layer,
	// so extent is 1
	virtual int get_delta_extent() = 0;
	virtual int get_gamma_extent() = 0;

	// Get the value of Δ[i, j, k]
	virtual iris_real get_delta(int i, int j, int k) = 0;
	virtual iris_real get_gamma(int i, int j, int k) = 0;

	void trace(const char *in_name);
	void trace2(const char *in_name);

    protected:
	iris_real m_hx;  // Δx
	iris_real m_hy;  // Δy
	iris_real m_hz;  // Δz
    };

}

#endif
