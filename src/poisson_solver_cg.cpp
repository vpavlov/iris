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
#include <cmath>
#include "poisson_solver_cg.h"
#include "mesh.h"
#include "timer.h"
#include "logger.h"
#include "memory.h"
#include "haloex.h"
#include "comm_rec.h"
#include "proc_grid.h"
#include "tags.h"

using namespace ORG_NCSA_IRIS;

poisson_solver_cg::poisson_solver_cg(class iris *obj)
    : poisson_solver(obj), m_stencil_width(3), m_max_iters(1000), m_epsilon(1.0e-5),
      m_phi(NULL), m_Ap(NULL), m_p(NULL), m_r(NULL), m_stencil(NULL),
      m_phi_haloex(NULL), m_p_haloex(NULL)
{
};

poisson_solver_cg::~poisson_solver_cg()
{
    memory::destroy_3d(m_stencil);
    memory::destroy_3d(m_phi);
    memory::destroy_3d(m_Ap);
    memory::destroy_3d(m_p);
    memory::destroy_3d(m_r);

    if(m_phi_haloex != NULL) {
	delete m_phi_haloex;
    }

    if(m_p_haloex != NULL) {
	delete m_p_haloex;
    }
}

void poisson_solver_cg::commit()
{
    if(!m_dirty) {
	return;
    }

    init_stencil();
    
    m_ext_size[0] = m_mesh->m_own_size[0] + m_stencil_width - 1;
    m_ext_size[1] = m_mesh->m_own_size[1] + m_stencil_width - 1;
    m_ext_size[2] = m_mesh->m_own_size[2] + m_stencil_width - 1;

    memory::destroy_3d(m_phi);
    memory::create_3d(m_phi, m_ext_size[0], m_ext_size[1], m_ext_size[2], false);

    axpby(0.0, m_mesh->m_rho, 1.0, m_mesh->m_rho, m_phi, false, false, true);

    memory::destroy_3d(m_p);
    memory::create_3d(m_p, m_ext_size[0], m_ext_size[1], m_ext_size[2]);

    memory::destroy_3d(m_Ap);
    memory::create_3d(m_Ap, m_mesh->m_own_size[0], m_mesh->m_own_size[1], m_mesh->m_own_size[2]);

    memory::destroy_3d(m_r);
    memory::create_3d(m_r, m_mesh->m_own_size[0], m_mesh->m_own_size[1], m_mesh->m_own_size[2]);

    if(m_phi_haloex != NULL) {
	delete m_phi_haloex;
    }

    m_phi_haloex = new haloex(m_local_comm->m_comm,
			      &(m_proc_grid->m_hood[0][0]),
			      1,
			      m_phi,
			      m_ext_size,
			      (m_stencil_width-1)/2,
			      (m_stencil_width-1)/2,
			      IRIS_TAG_PHI_HALO);

    if(m_p_haloex != NULL) {
	delete m_p_haloex;
    }

    m_p_haloex = new haloex(m_local_comm->m_comm,
			    &(m_proc_grid->m_hood[0][0]),
			    1,
			    m_p,
			    m_ext_size,
			    (m_stencil_width-1)/2,
			    (m_stencil_width-1)/2,
			    IRIS_TAG_P_HALO);
    
    m_dirty = false;
}

void poisson_solver_cg::init_stencil()
{
    switch(m_stencil_width) {
    case 3:
	init_stencil3();
	break;

    default:
	throw std::logic_error("Stencil width not implemented!");
    }
}

void poisson_solver_cg::init_stencil3()
{
    memory::destroy_3d(m_stencil);
    memory::create_3d(m_stencil, 3, 3, 3, true);
    m_stencil[1][1][1] = -6;
    m_stencil[0][1][1] = 1;
    m_stencil[1][0][1] = 1;
    m_stencil[1][1][0] = 1;
    m_stencil[2][1][1] = 1;
    m_stencil[1][2][1] = 1;
    m_stencil[1][1][2] = 1;
}

void poisson_solver_cg::adot(iris_real ***in, iris_real ***out, haloex *hex)
{
    int sw = (m_stencil_width - 1)/2;

    hex->exch_full();

    for(int i=0;i<m_mesh->m_own_size[0];i++) {
	for(int j=0;j<m_mesh->m_own_size[1];j++) {
	    for(int k=0;k<m_mesh->m_own_size[2];k++) {
		out[i][j][k] = (iris_real)0.0;
		for(int ii=-sw;ii<=sw;ii++) {
		    for(int jj=-sw;jj<=sw;jj++) {
			for(int kk=-sw;kk<=sw;kk++) {
			    out[i][j][k] += in[i+ii+sw][j+jj+sw][k+kk+sw] * m_stencil[ii+sw][jj+sw][kk+sw];
			}
		    }
		}
	    }
	}
    }
}

iris_real poisson_solver_cg::dot(iris_real ***v1, iris_real ***v2, bool v1_has_halo, bool v2_has_halo)
{
    iris_real sum = 0.0;
    iris_real retval;
    
    int v1s, v2s;

    if(v1_has_halo) {
	v1s = (m_stencil_width - 1)/2;
    }else {
	v1s = 0;
    }

    if(v2_has_halo) {
	v2s = (m_stencil_width - 1)/2;
    }else {
	v2s = 0;
    }

    int ex, ey, ez;
    ex = m_mesh->m_own_size[0];
    ey = m_mesh->m_own_size[1];
    ez = m_mesh->m_own_size[2];

    for(int i=0;i<ex;i++) {
	for(int j=0;j<ey;j++) {
	    for(int k=0;k<ez;k++) {
		sum += v1[i+v1s][j+v1s][k+v1s] * v2[i+v2s][j+v2s][k+v2s];
	    }
	}
    }

    MPI_Allreduce(&sum, &retval, 1, IRIS_REAL, MPI_SUM, m_local_comm->m_comm);
    return retval;
}

void poisson_solver_cg::axpby(iris_real a, iris_real ***x, iris_real b, iris_real ***y, iris_real ***out,
			      bool x_has_halo, bool y_has_halo, bool out_has_halo)
{
    int xs, ys, outs;
    if(x_has_halo) {
	xs = (m_stencil_width - 1)/2;
    }else {
	xs = 0;
    }

    if(y_has_halo) {
	ys = (m_stencil_width - 1)/2;
    }else {
	ys = 0;
    }

    if(out_has_halo) {
	outs = (m_stencil_width - 1)/2;
    }else {
	outs = 0;
    }

    int ex, ey, ez;
    ex = m_mesh->m_own_size[0];
    ey = m_mesh->m_own_size[1];
    ez = m_mesh->m_own_size[2];

    for(int i=0;i<ex;i++) {
	for(int j=0;j<ey;j++) {
	    for(int k=0;k<ez;k++) {
		out[i+outs][j+outs][k+outs] = a * x[i+xs][j+xs][k+xs] + b * y[i+ys][j+ys][k+ys];
	    }
	}
    }
}

void poisson_solver_cg::solve()
{
    timer tm;

    tm.start();
    m_mesh->convolve_with_gaussian();
    tm.stop();
    m_logger->info("Convolution wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(),
		   (tm.read_cpu() * 100.0) /tm.read_wall());

    // r = rho - Adot(phi,xsize,ysize,zsize)
    adot(m_phi, m_Ap, m_phi_haloex);
    axpby(1.0, m_mesh->m_rho, -1.0, m_Ap, m_r, false, false, false);

    // p = r  // TODO: write a real function for this instead of butchering the machine
    axpby(0.0, m_r, 1.0, m_r, m_p, false, false, true);

    // for (kk in 1:maxits) {
    for(int i=0;i<m_max_iters;i++) {
	//   alpha = sum(r * r) / sum(p * Adot(p,xsize,ysize,zsize))
	//   and also several lines later normr2old = sum(r * r)
	iris_real rr_old = dot(m_r, m_r, false, false);
	adot(m_p, m_Ap, m_p_haloex);
	iris_real pap = dot(m_p, m_Ap, true, false);
	iris_real alpha = rr_old/pap;
	
	//   phi = phi + alpha * p
	axpby(1.0, m_phi, alpha, m_p, m_phi, true, true, true);

	//   r = r - alpha * (Adot(p,xsize,ysize,zsize))
	axpby(1.0, m_r, -alpha, m_Ap, m_r, false, false, false);

	//   normr2 = sum(r * r)
	iris_real rr = dot(m_r, m_r, false, false);
	iris_real phiphi = dot(m_phi, m_phi, true, true);
	iris_real err = sqrt(rr)/sqrt(phiphi);

	//   # print(paste("it",kk,"|r|", sqrt(normr2)))
	m_logger->trace("cg it %d: |r|/|phi| = %f", i, err);

	//   if (normr2 <= epsilon ** 2) {
	//     print(paste("|r|", sqrt(normr2)))
	//     break
	//   }
	if(err <= m_epsilon) {
	    m_logger->trace("cg done");
	    break;
	}

	//   beta = normr2 / normr2old
	iris_real beta = rr / rr_old;

	//   p = r + beta * p
	axpby(1.0, m_r, beta, m_p, m_p, false, true, true);
    }

    axpby(0.0, m_phi, 1.0, m_phi, m_mesh->m_phi, true, true, false);
}
