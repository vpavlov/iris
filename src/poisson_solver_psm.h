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

	void dump_rho(char *fname);
	void dump_rho2(char *fname);

    private:
	void calculate_eigenvalues();
	void setup_fft_grid();
	void setup_remap();

	void copy_rho_from_mesh();

    private:
	// our version of domain decomposition, whatever's most suitable for
	// this kind of solver.
	class fft_grid *m_fft_grid;
	int             m_own_size[3];
	int             m_own_offset[3];

	// 3D Array of stencil eigenvalues (local portion only).
	// Has the same dimension as the local mesh.
	iris_real ***m_ev;
	iris_real *m_rho;  // our version of rho, used for fft

	class remap *m_remap;
	iris_real *m_scratch;  // scratch space for remapping
    };
}

#endif
