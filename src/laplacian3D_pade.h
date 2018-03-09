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
#ifndef __IRIS_LAPLACIAN3D_PADE_H__
#define __IRIS_LAPLACIAN3D_PADE_H__

#include "laplacian3D.h"
#include "real.h"

namespace ORG_NCSA_IRIS {

    //
    // A symmetrical stencil that approximates a 3D Laplacian:
    //
    // d2/dx^2 + d2/dy^2 + d2/dz^2
    //
    // using Pade approximation
    class laplacian3D_pade : public laplacian3D {
	
    public:
	laplacian3D_pade(int in_m, int in_n, bool in_cut);
	laplacian3D_pade(int in_m, int in_n, bool in_cut,
			 iris_real in_hx, iris_real in_hy, iris_real in_hz);
	~laplacian3D_pade();

	void commit();

	int get_delta_extent();
	int get_gamma_extent();

	iris_real get_delta(int i, int j, int k);
	iris_real get_gamma(int i, int j, int k);

    private:

	void compute_rhs(iris_real *denom);
	void compute_lhs(iris_real *nom, iris_real *denom);

	int m_m;  // The upper order of the Pade approximant P[m, n]
	int m_n;  // The lower order of the Pade approximant P[m, n]
	bool m_cut;  // Whether to cut to the desired accuracy or use all terms
    };

}

#endif
