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
#include <stdexcept>
#include "laplacian3D_pade.h"
#include "memory.h"
#include "utils.h"
#include "cdo3D.h"

using namespace ORG_NCSA_IRIS;

// 3D Laplacian approximation, based on Pade approximants of the Taylor
// expansion of 
//
// (hD)^2 = 4*asinh(δ/2)^2
//
// See iris/lisp/stencil_coeff.lisp for explanation of the theory behind this


// The first several terms (19 in this case) of the Taylor expansion.
//
// These should be more than enough, since they cover accuracy order of at
// least up to h^20, which in case h = 1/128 is O(10^-43). If however, this is
// somehow not enough, load the above mentioned lisp file and execute
// (d2/dx2-taylor coeff 100) or something. Do note that it gives only odd
// (non-zero) coeffs.
//
static iris_real taylor_coeff[] = {
    1.0, 0.0, -1.0/12, 0.0, 1.0/90, 0.0, -1.0/560, 0.0, 1.0/3150, 0.0,
    -1.0/16632, 0.0, 1.0/84084, 0.0, -1.0/411840, 0.0, 1.0/1969110, 0.0,
    -1.0/9237800
};

laplacian3D_pade::laplacian3D_pade(int in_m, int in_n, bool in_cut)
    : laplacian3D(in_m + in_n + 2)  // P[m, n] is of acc. order m+n+2
{
    m_m = in_m;
    m_n = in_n;
    m_cut = in_cut;
}

laplacian3D_pade::laplacian3D_pade(int in_m, int in_n, bool in_cut,
				   iris_real in_hx,
				   iris_real in_hy, iris_real in_hz)
    : laplacian3D_pade(in_m, in_n, in_cut)
{
    set_hx(in_hx);
    set_hy(in_hy);
    set_hz(in_hz);
}

laplacian3D_pade::~laplacian3D_pade()
{
    if(m_delta != NULL) {
	cdo3D *data = (cdo3D *)m_delta;
	delete data;
    }

    if(m_gamma != NULL) {
	cdo3D *data = (cdo3D *)m_gamma;
	delete data;
    }
}

void laplacian3D_pade::commit()
{
    if(!m_dirty) {
	return;
    }

    iris_real *nom, *denom;
    pade_approximant(m_m, m_n, taylor_coeff, nom, denom);
    compute_rhs(denom);
    compute_lhs(nom, denom);
}

void laplacian3D_pade::compute_rhs(iris_real *denom)
{
    cdo3D *retval = new cdo3D(m_n, 0.0, 0, 0, 0);

    for(int i=0;i<=m_n;i+=2) {     // even elements (odd powers of δ) are 0
	for(int j=0;j<=m_n;j+=2) {
	    for(int k=0;k<=m_n;k+=2) {

		// NOTE:
		// The paper
		// 
		// "High-order compact solvers for the three-dimensional
		// Poisson equation" by Godehard Sutmann, Bernhard Steffen
		// 
		// on page 147 says "To have the RHS of fourth order, it is
		// necessary to keep the first two terms only",
		// which leaves out terms like δ^2x * δ^2y, which is obviously
		// fourth order. Meanwhile, they keep the similar terms for the
		// LHS. I don't see how this is correct, will have to talk to
		// Godehard next time...
		// [VNP]
		if(m_cut && (i+j+k > m_acc)) {
		    continue;
		}

		cdo3D next(m_n, denom[i]*denom[j]*denom[k], i, j, k);
		(*retval) += next;
	    }
	}
    }
    m_gamma = (void *)retval;
    m_lhs_only = (m_n == 0);
}

void laplacian3D_pade::compute_lhs(iris_real *nom, iris_real *denom)
{
    iris_real hx2 = m_hx * m_hx;
    iris_real hy2 = m_hy * m_hy;
    iris_real hz2 = m_hz * m_hz;
    int ord = MAX(m_m+2, m_n);
    cdo3D *retval = new cdo3D(ord, 0.0, 0, 0, 0);

    // nom x * denom y * denom z
    for(int i=0;i<=m_m;i+=2) {     // even elements (odd powers of δ) are 0
	for(int j=0;j<=m_n;j+=2) {
	    for(int k=0;k<=m_n;k+=2) {

		if(m_cut && (i+j+k+2 > m_acc)) {
		    continue;
		}

		cdo3D next(ord, nom[i]*denom[j]*denom[k] / hx2, i+2, j, k);
		(*retval) += next;
	    }
	}
    }

    // denom x * nom y * denom z
    for(int i=0;i<=m_n;i+=2) {     // even elements (odd powers of δ) are 0
	for(int j=0;j<=m_m;j+=2) {
	    for(int k=0;k<=m_n;k+=2) {

		if(m_cut && (i+j+k+2 > m_acc)) {
		    continue;
		}

		cdo3D next(ord, denom[i]*nom[j]*denom[k] / hy2, i, j+2, k);
		(*retval) += next;
	    }
	}
    }

    // denom x * denom y * nom z
    for(int i=0;i<=m_n;i+=2) {     // even elements (odd powers of δ) are 0
	for(int j=0;j<=m_n;j+=2) {
	    for(int k=0;k<=m_m;k+=2) {

		if(m_cut && (i+j+k+2 > m_acc)) {
		    continue;
		}

		cdo3D next(ord, denom[i]*denom[j]*nom[k] / hz2, i, j, k+2);
		(*retval) += next;
	    }
	}
    }
    m_delta = (void *)retval;
}

inline iris_real laplacian3D_pade::get_delta(int i, int j, int k)
{
    return ((cdo3D *)m_delta)->m_data[i][j][k];
}

inline iris_real laplacian3D_pade::get_gamma(int i, int j, int k)
{
    return ((cdo3D *)m_gamma)->m_data[i][j][k];
}

inline int laplacian3D_pade::get_delta_extent() 
{
    return ((cdo3D *)m_delta)->m_n/2;
};

inline int laplacian3D_pade::get_gamma_extent()
{
    return ((cdo3D *)m_gamma)->m_n/2;
};
