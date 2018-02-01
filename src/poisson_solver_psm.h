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
#ifndef __IRIS_POISSON_SOLVER_PSM_H__
#define __IRIS_POISSON_SOLVER_PSM_H__

#include "poisson_solver.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver_psm : public poisson_solver {

    public:
	poisson_solver_psm(class iris *obj);
	~poisson_solver_psm();

	void commit();
	void solve();

    private:
	void calculate_laplacian_ev();
	void calculate_ddx_ev(int idx, class first_derivative *ddx,
			      iris_real *&ddx_ev);

	// Calculate the image of the potential (φ) in reciprocal (k) space
	// Basically, divide the image of ρ through the eigenvalues of the
	// Laplacian.
	// It does so in-place
	void kspace_phi(iris_real *io_rho_phi);
	void kspace_Ex(iris_real *in_phi, iris_real *out_Ex);
	void kspace_Ey(iris_real *in_phi, iris_real *out_Ey);
	void kspace_Ez(iris_real *in_phi, iris_real *out_Ez);

	void dump_work(int i);

    private:

	iris_real ***m_ev;    // 3D array of laplacian engeinvalues
	iris_real *m_ddx_ev;  // 1D array of d/dx stencil eigenvalues
	iris_real *m_ddy_ev;  // 1D array of d/dy stencil eigenvalues
	iris_real *m_ddz_ev;  // 1D array of d/dz stencil eigenvalues

	class fft3d *m_fft;

	// FFT workspaces
	iris_real *m_work1;
	iris_real *m_work2;
    };
}

#endif
