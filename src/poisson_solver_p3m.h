// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2019, the National Center for Supercomputing Applications
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
#ifndef __IRIS_POISSON_SOLVER_P3M_H__
#define __IRIS_POISSON_SOLVER_P3M_H__

#include "charge_assigner.h"
#include "poisson_solver.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver_p3m : public poisson_solver {

    public:
	poisson_solver_p3m(class iris *obj);
	~poisson_solver_p3m();

	void commit();
	void solve();

    private:
	void kspace_phi(iris_real *io_rho_phi);
	void kspace_eng(iris_real *in_rho_phi);
	void kspace_Ex(iris_real *in_phi, iris_real *out_Ex);
	void kspace_Ey(iris_real *in_phi, iris_real *out_Ey);
	void kspace_Ez(iris_real *in_phi, iris_real *out_Ez);

	void calculate_green_function();
	void calculate_k();
	void calculate_virial_coeff();
        void calculate_denominator();

	inline iris_real denominator(const iris_real &x, const iris_real &y, const iris_real &z)
	{
	    iris_real sx, sy, sz;
	    sx = sy = sz = 0.0;

	    for(int i = m_chass->m_order - 1; i >= 0; i--) {
		iris_real c = m_chass->m_gfd_coeff[i];
		sx = c + sx * x;
		sy = c + sy * y;
		sz = c + sz * z;
	    }
	    iris_real s = sx * sy * sz;
	    return s*s;
	}

    private:
	iris_real ***m_greenfn;  // green function table, actually a 3D array
	iris_real ***m_denominator; // denominators buffer
	iris_real *m_kx;         // kx (2πi/L) tabulated
	iris_real *m_ky;         // ky (2πj/L) tabulated
	iris_real *m_kz;         // kz (2πk/L) tabulated
	iris_real **m_vc;        // virial coefficients tabulated, 2D array [nx*ny*nz][6]

	int m_fft_size[3];
	int m_fft_offset[3];
	class grid  *m_fft_grid;
	class remap *m_remap;
	class fft3d *m_fft1, *m_fft2;
      
	// FFT workspaces
	iris_real *m_work1;
	iris_real *m_work2;
	iris_real *m_work3;  // temporary, to be removed
    };
}

#endif
