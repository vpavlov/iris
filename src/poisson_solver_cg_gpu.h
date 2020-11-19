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
#ifndef __IRIS_POISSON_SOLVER_CG_H__
#define __IRIS_POISSON_SOLVER_CG_H__

#include "poisson_solver_gpu.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver_cg_gpu : public poisson_solver_gpu {

    public:
	poisson_solver_cg_gpu(class iris_gpu *obj);
	~poisson_solver_cg_gpu();

	void commit();
	void solve();
	void handle_box_resize() {};
	
	void set_stencil_width(int in_width);
	void set_max_iters(int in_max_iters);
	void set_epsilon(iris_real in_epsilon);

    private:
	// convolution-related helpers
	void init_convolution();
	void convolve_with_gaussian();
	void prepare_for_gx();
	void prepare_for_gx_test();
	void prepare_for_gy();
	void prepare_for_gz();
	void add_gx();
	void add_gy();
	void add_gz();
	void extract_rho();

	void init_stencil();

	void blur_rhs();

	void adot(iris_real ***in, iris_real ***out, class haloex_gpu *hex);
	iris_real dot(iris_real ***v1, iris_real ***v2, bool v1_has_halo, bool v2_has_halo);
	void axpby(iris_real a, iris_real ***x, iris_real b, iris_real ***y, iris_real ***out,
		   bool x_has_halo, bool y_has_halo, bool out_has_halo);

	// config
	iris_real m_nsigmas;  // total width of Gaussian in # of σ's

	class laplacian3D *m_stencil;

	int m_max_iters;
	iris_real m_epsilon;


	// convolution-related fields
	int           m_gauss_width[3];  // half-width of Gaussian in # of cells
	int           m_ext2_size[3];    // size of mesh + Gaussian halo
	iris_real  ***m_conv1;           // workspace for own ρ + gaussian halo
	iris_real  ***m_conv2;           // workspace for own ρ + gaussian halo
	class haloex_gpu *m_Gx_haloex;       // Gaussian halo exchanger, X dir
	class haloex_gpu *m_Gy_haloex;       // Gaussian halo exchanger, Y dir
	class haloex_gpu *m_Gz_haloex;       // Gaussian halo exchanger, Z dir

	int m_ext_size[3];     // size of own mesh (phi) + halo
	int m_rext_size[3];    // size of own mesh (rho) + halo
	
	iris_real ***m_rho;   // ρ - the right hand side; needs halo ?
	iris_real ***m_blurred_rho;   // ρ - the right hand side; needs halo ?
	iris_real ***m_phi;   // φ - the result; needs halo
	iris_real ***m_Ap;    // no need for halo
	iris_real ***m_p;     // needs halo
	iris_real ***m_r;     // no need for halo

	class haloex_gpu *m_phi_haloex;
	class haloex_gpu *m_p_haloex;
	class haloex_gpu *m_rho_haloex;
    };
}

#endif
