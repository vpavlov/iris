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
#ifndef __IRIS_POISSON_SOLVER_H__
#define __IRIS_POISSON_SOLVER_H__

#include "state_accessor.h"
#include "charge_assigner.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver : protected state_accessor {

    public:
	poisson_solver(class iris *obj);
	~poisson_solver();

	void commit();
	void solve();

	void set_dirty(bool in_dirty) { m_dirty = in_dirty; };
	
    private:
	void kspace_phi(iris_real *io_rho_phi);
	void kspace_eng(iris_real *in_rho_phi);
	void kspace_Ex(iris_real *in_phi, iris_real *out_Ex);
	void kspace_Ey(iris_real *in_phi, iris_real *out_Ey);
	void kspace_Ez(iris_real *in_phi, iris_real *out_Ez);

	void calculate_green_function();
	void calculate_k();

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
	bool m_dirty;  // wether to recalculate on commit
	iris_real ***m_greenfn;  // green function table, actually a 3D array
	iris_real *m_kx;
	iris_real *m_ky;
	iris_real *m_kz;
	class fft3d *m_fft;

	// FFT workspaces
	iris_real *m_work1;
	iris_real *m_work2;
	iris_real *m_work3;  // temporary, to be removed
    };
}

#endif
