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
#include "solver_param.h"
#include "laplacian3D_pade.h"

using namespace ORG_NCSA_IRIS;

#define _PI2    1.5707963267948966
#define _SQRT_2 1.4142135623730951

#define _DEFAULT_NSIGMAS 6.0  // default total width of Gaussian in # of σ's

poisson_solver_cg::poisson_solver_cg(class iris *obj)
    : poisson_solver(obj), m_nsigmas(0), m_gauss_width{0, 0, 0},
      m_ext2_size{0, 0, 0}, m_conv1(NULL), m_conv2(NULL),
      m_Gx_haloex(NULL), m_Gy_haloex(NULL), m_Gz_haloex(NULL),
      m_stencil(NULL), 
      m_max_iters(10000), m_epsilon(1.0e-5),
      m_phi(NULL), m_rho(NULL), m_blurred_rho(NULL), m_Ap(NULL), m_p(NULL), m_r(NULL),
      m_phi_haloex(NULL), m_p_haloex(NULL), m_rho_haloex(NULL)
{
};

poisson_solver_cg::~poisson_solver_cg()
{
    memory::destroy_3d(m_conv1);
    memory::destroy_3d(m_conv2);
    if(m_Gx_haloex != NULL) { delete m_Gx_haloex; }
    if(m_Gy_haloex != NULL) { delete m_Gy_haloex; }
    if(m_Gz_haloex != NULL) { delete m_Gz_haloex; }

    if(m_stencil != NULL) { delete m_stencil; }

    memory::destroy_3d(m_phi);
    memory::destroy_3d(m_rho);
    memory::destroy_3d(m_blurred_rho);
    memory::destroy_3d(m_Ap);
    memory::destroy_3d(m_p);
    memory::destroy_3d(m_r);

    if(m_phi_haloex != NULL) { delete m_phi_haloex; }
    if(m_rho_haloex != NULL) { delete m_rho_haloex; }
    if(m_p_haloex != NULL) { delete m_p_haloex; }
}

void poisson_solver_cg::init_convolution()
{
    solver_param_t p = m_iris->get_solver_param(IRIS_SOLVER_CG_NSIGMAS);
    m_nsigmas = p.r;
    if(m_nsigmas == 0.0) {
	m_nsigmas = _DEFAULT_NSIGMAS;
    }
    
    iris_real s = 1.0/(sqrt(2) * m_iris->m_alpha);  // σ = 1/α*sqrt(2)
    
    m_gauss_width[0] = ceil(0.5 * m_nsigmas * s * m_mesh->m_hinv[0]);
    m_gauss_width[1] = ceil(0.5 * m_nsigmas * s * m_mesh->m_hinv[1]);
    m_gauss_width[2] = ceil(0.5 * m_nsigmas * s * m_mesh->m_hinv[2]);

    m_logger->trace("CG Solver: Gaussian σ = %f %s", s, m_units->length_unit);
    m_logger->trace("CG Solver: Using %.2fσ = %f %s to approximate the Gaussian", m_nsigmas, m_nsigmas * s, m_units->length_unit);
    m_logger->trace("CG Solver: Gaussian is %dx%dx%d", m_gauss_width[0]*2,
		    m_gauss_width[1]*2, m_gauss_width[2]*2);

    m_ext2_size[0] = m_mesh->m_own_size[0] + m_gauss_width[0]*2;
    m_ext2_size[1] = m_mesh->m_own_size[1] + m_gauss_width[1]*2;
    m_ext2_size[2] = m_mesh->m_own_size[2] + m_gauss_width[2]*2;

    memory::destroy_3d(m_conv1);
    memory::destroy_3d(m_conv2);
    if(m_Gx_haloex != NULL) { delete m_Gx_haloex; }
    if(m_Gy_haloex != NULL) { delete m_Gy_haloex; }
    if(m_Gz_haloex != NULL) { delete m_Gz_haloex; }

    memory::create_3d(m_conv1, m_ext2_size[0], m_ext2_size[1], m_ext2_size[2]);
    memory::create_3d(m_conv2, m_ext2_size[0], m_ext2_size[1], m_ext2_size[2]);

    m_Gx_haloex = new haloex(m_local_comm->m_comm,
			     &(m_proc_grid->m_hood[0][0]),
			     1,
			     m_conv1,
			     m_ext2_size,
			     m_gauss_width[0],
			     m_gauss_width[0],
			     IRIS_TAG_GX_HALO);
    
    m_Gy_haloex = new haloex(m_local_comm->m_comm,
			     &(m_proc_grid->m_hood[0][0]),
			     1,
			     m_conv2,
			     m_ext2_size,
			     m_gauss_width[1],
			     m_gauss_width[1],
			     IRIS_TAG_GY_HALO);
    
    m_Gz_haloex = new haloex(m_local_comm->m_comm,
			     &(m_proc_grid->m_hood[0][0]),
			     1,
			     m_conv1,
			     m_ext2_size,
			     m_gauss_width[2],
			     m_gauss_width[2],
			     IRIS_TAG_GZ_HALO);
}

void poisson_solver_cg::commit()
{
    if(!m_dirty) {
	return;
    }

    init_convolution();
    init_stencil();
    
    int sw = m_stencil->get_delta_extent();
    m_ext_size[0] = m_mesh->m_own_size[0] + sw*2;
    m_ext_size[1] = m_mesh->m_own_size[1] + sw*2;
    m_ext_size[2] = m_mesh->m_own_size[2] + sw*2;

    int swg = m_stencil->get_gamma_extent();
    m_rext_size[0] = m_mesh->m_own_size[0] + swg*2;
    m_rext_size[1] = m_mesh->m_own_size[1] + swg*2;
    m_rext_size[2] = m_mesh->m_own_size[2] + swg*2;
    
    memory::destroy_3d(m_phi);
    memory::create_3d(m_phi, m_ext_size[0], m_ext_size[1], m_ext_size[2], false);

    memory::destroy_3d(m_rho);
    memory::create_3d(m_rho, m_rext_size[0], m_rext_size[1], m_rext_size[2], true);

    memory::destroy_3d(m_blurred_rho);
    memory::create_3d(m_blurred_rho, m_mesh->m_own_size[0], m_mesh->m_own_size[1], m_mesh->m_own_size[2], true);
    
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
			      sw,
			      sw,
			      IRIS_TAG_PHI_HALO);

    if(m_rho_haloex != NULL) {
	delete m_rho_haloex;
    }

    m_rho_haloex = new haloex(m_local_comm->m_comm,
			      &(m_proc_grid->m_hood[0][0]),
			      1,
			      m_rho,
			      m_rext_size,
			      swg,
			      swg,
			      IRIS_TAG_RHO_HALO);
    
    if(m_p_haloex != NULL) {
	delete m_p_haloex;
    }

    m_p_haloex = new haloex(m_local_comm->m_comm,
			    &(m_proc_grid->m_hood[0][0]),
			    1,
			    m_p,
			    m_ext_size,
			    sw,
			    sw,
			    IRIS_TAG_P_HALO);
    
    m_dirty = false;
}

void poisson_solver_cg::init_stencil()
{
    solver_param_t p = m_iris->get_solver_param(IRIS_SOLVER_CG_STENCIL_PADE_M);
    int m = p.i;

    p = m_iris->get_solver_param(IRIS_SOLVER_CG_STENCIL_PADE_N);
    int n = p.i;

    m_stencil = new laplacian3D_pade(m, n, false, m_mesh->m_h[0],
				     m_mesh->m_h[1], m_mesh->m_h[2]);
    m_stencil->commit();
    m_stencil->trace2("XX");
}

void poisson_solver_cg::prepare_for_gx()
{
    int x1 = m_gauss_width[0];
    int y1 = m_gauss_width[1];
    int z1 = m_gauss_width[2];

    int x2 = x1 + m_mesh->m_own_size[0];
    int y2 = y1 + m_mesh->m_own_size[1];
    int z2 = z1 + m_mesh->m_own_size[2];

    // NOTE: This is made in 3 distinct loops so there is are no ifs in them
    //       which would otherwise break auto-vectorization

    // first m_gauss_width X planes are all 0
    for(int i=0;i<x1;i++) {
	for(int j=y1;j<y2;j++) {
	    for(int k=z1;k<z2;k++) {
		m_conv1[i][j][k] = (iris_real)0.0;
	    }
	}
    }

    // m_own_size[0] X planes are copied from ρ
    for(int i=x1;i<x2;i++) {
	for(int j=y1;j<y2;j++) {
	    for(int k=z1;k<z2;k++) {
		m_conv1[i][j][k] = m_mesh->m_rho[i-x1][j-y1][k-z1];
	    }
	}
    }

    // last m_gauss_width X planes are all 0
    for(int i=x2;i<m_ext2_size[0];i++) {
	for(int j=y1;j<y2;j++) {
	    for(int k=z1;k<z2;k++) {
		m_conv1[i][j][k] = (iris_real)0.0;
	    }
	}
    }

    // Now exchange the halo in X direction
    m_Gx_haloex->exch_x();

    memset(&(m_conv2[0][0][0]), 0,
	   m_ext2_size[0] * m_ext2_size[1] * m_ext2_size[2] *
	   sizeof(iris_real));
}

void poisson_solver_cg::prepare_for_gx_test()
{
    for(int i=0;i<m_ext2_size[0];i++) {
	for(int j=0;j<m_ext2_size[1];j++) {
	    for(int k=0;k<m_ext2_size[2];k++) {
		m_conv1[i][j][k] = (iris_real)0.0;
	    }
	}
    }

    m_conv1[28][28][28] = (iris_real)1.0 / (m_mesh->m_h[0] * m_mesh->m_h[1] * m_mesh->m_h[2]);

    // Now exchange the halo in X direction
    m_Gx_haloex->exch_x();

    memset(&(m_conv2[0][0][0]), 0,
	   m_ext2_size[0] * m_ext2_size[1] * m_ext2_size[2] *
	   sizeof(iris_real));
}

void poisson_solver_cg::add_gx()
{
    iris_real a = m_iris->m_alpha;

    iris_real h = m_mesh->m_h[0];

    int sx = m_gauss_width[0];
    int ex = sx + m_mesh->m_own_size[0];

    int sy = m_gauss_width[1];
    int ey = sy + m_mesh->m_own_size[1];

    int sz = m_gauss_width[2];
    int ez = sz + m_mesh->m_own_size[2];

    iris_real t2 = a / sqrt(M_PI);
    
    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		for(int m=-sx;m<=sx;m++) {
		    iris_real q = m_conv1[i+m][j][k];
		    if(q == 0.0) {
			continue;
		    }

		    iris_real xa_x0 = -(m*h + h/2.0);
		    iris_real xb_x0 = xa_x0 + h;
		    iris_real x_x0 = m*h;

		    if(m == -sx) {
		    	xb_x0 = 1000*h;
		    }
		    if(m == sx) {
		    	xa_x0 = -1000*h;
		    }

		    //iris_real v = erf(a * xb_x0) - erf(a * xa_x0);

		    iris_real v = 2 * h * t2 * exp(-1.0*a*a*(x_x0)*(x_x0));

		    // q is in fact q/h^3 - see mesh::assign_charges1
		    m_conv2[i][j][k] += q * v;
		}
	    }
	}
    }
}

void poisson_solver_cg::prepare_for_gy()
{
    m_Gy_haloex->exch_y();
    
    memset(&(m_conv1[0][0][0]), 0,
	   m_ext2_size[0] * m_ext2_size[1] * m_ext2_size[2] *
	   sizeof(iris_real));
}


void poisson_solver_cg::add_gy()
{
    iris_real a = m_iris->m_alpha;

    iris_real h = m_mesh->m_h[1];

    int sx = m_gauss_width[0];
    int ex = sx + m_mesh->m_own_size[0];

    int sy = m_gauss_width[1];
    int ey = sy + m_mesh->m_own_size[1];

    int sz = m_gauss_width[2];
    int ez = sz + m_mesh->m_own_size[2];

    iris_real t2 = a / sqrt(M_PI);

    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		for(int m=-sy;m<=sy;m++) {
		    iris_real q = m_conv2[i][j+m][k];
		    if(q == 0.0) {
			continue;
		    }

		    iris_real xa_x0 = -(m*h + h/2.0);
		    iris_real xb_x0 = xa_x0 + h;
		    iris_real x_x0 = m*h;

		    if(m == -sx) {
		    	xb_x0 = 1000*h;
		    }
		    if(m == sx) {
		    	xa_x0 = -1000*h;
		    }

		    //iris_real v = erf(a * xb_x0) - erf(a * xa_x0);

		    iris_real v = 2 * h * t2 * exp(-1.0*a*a*(x_x0)*(x_x0));
		    
		    m_conv1[i][j][k] += q * v;
		}
	    }
	}
    }
}

void poisson_solver_cg::prepare_for_gz()
{
    m_Gz_haloex->exch_z();

    memset(&(m_conv2[0][0][0]), 0,
	   m_ext2_size[0] * m_ext2_size[1] * m_ext2_size[2] *
	   sizeof(iris_real));
}


void poisson_solver_cg::add_gz()
{
    iris_real a = m_iris->m_alpha;

    iris_real h = m_mesh->m_h[2];

    int sx = m_gauss_width[0];
    int ex = sx + m_mesh->m_own_size[0];

    int sy = m_gauss_width[1];
    int ey = sy + m_mesh->m_own_size[1];

    int sz = m_gauss_width[2];
    int ez = sz + m_mesh->m_own_size[2];

    iris_real t2 = a / sqrt(M_PI);
    
    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		for(int m=-sz;m<=sz;m++) {
		    iris_real q = m_conv1[i][j][k+m];
		    if(q == 0.0) {
			continue;
		    }

		    iris_real xa_x0 = -(m*h + h/2.0);
		    iris_real xb_x0 = xa_x0 + h;
		    iris_real x_x0 = m*h;

		    if(m == -sx) {
		    	xb_x0 = 1000*h;
		    }
		    if(m == sx) {
		    	xa_x0 = -1000*h;
		    }

		    //iris_real v = erf(a * xb_x0) - erf(a * xa_x0);

		    iris_real v = 2 * h * t2 * exp(-1.0*a*a*(x_x0)*(x_x0));
		    
		    m_conv2[i][j][k] += q * v;
		}
	    }
	}
    }
}

void poisson_solver_cg::extract_rho()
{
    iris_real sum = 0.0;

    int off = m_stencil->get_gamma_extent();

    int sx = m_gauss_width[0];
    int ex = sx + m_mesh->m_own_size[0];

    int sy = m_gauss_width[1];
    int ey = sy + m_mesh->m_own_size[1];

    int sz = m_gauss_width[2];
    int ez = sz + m_mesh->m_own_size[2];

    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		sum += m_conv2[i][j][k];
		m_rho[off+i-sx][off+j-sy][off+k-sz] = -_PI2 * m_conv2[i][j][k];
		//printf("%g\n", i-sx, j-sy, k-sz, m_rho[off+i-sx][off+j-sy][off+k-sz]);
	    }
	}
    }

    m_logger->info("RHO SUM = %f", sum);
}

void poisson_solver_cg::convolve_with_gaussian()
{
    prepare_for_gx();
    add_gx();

    prepare_for_gy();
    add_gy();

    prepare_for_gz();
    add_gz();

    extract_rho();
}

void poisson_solver_cg::adot(iris_real ***in, iris_real ***out, haloex *hex)
{
    int sw = m_stencil->get_delta_extent();

    hex->exch_full();

    for(int i=0;i<m_mesh->m_own_size[0];i++) {
	for(int j=0;j<m_mesh->m_own_size[1];j++) {
	    for(int k=0;k<m_mesh->m_own_size[2];k++) {
		out[i][j][k] = (iris_real)0.0;
		for(int ii=-sw;ii<=sw;ii++) {
		    for(int jj=-sw;jj<=sw;jj++) {
			for(int kk=-sw;kk<=sw;kk++) {
			    out[i][j][k] += in[i+ii+sw][j+jj+sw][k+kk+sw] * m_stencil->get_delta(ii+sw, jj+sw, kk+sw);
			}
		    }
		}
	    }
	}
    }
}

void poisson_solver_cg::blur_rhs()
{
    int sw = m_stencil->get_gamma_extent();

    m_rho_haloex->exch_full();

    for(int i=0;i<m_mesh->m_own_size[0];i++) {
	for(int j=0;j<m_mesh->m_own_size[1];j++) {
	    for(int k=0;k<m_mesh->m_own_size[2];k++) {
		for(int ii=-sw;ii<=sw;ii++) {
		    for(int jj=-sw;jj<=sw;jj++) {
			for(int kk=-sw;kk<=sw;kk++) {
			    m_blurred_rho[i][j][k] += m_rho[i+ii+sw][j+jj+sw][k+kk+sw] * m_stencil->get_gamma(ii+sw, jj+sw, kk+sw);
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
	v1s = m_stencil->get_delta_extent();
    }else {
	v1s = 0;
    }

    if(v2_has_halo) {
	v2s = m_stencil->get_delta_extent();
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
	xs = m_stencil->get_delta_extent();
    }else {
	xs = 0;
    }

    if(y_has_halo) {
	ys = m_stencil->get_delta_extent();
    }else {
	ys = 0;
    }

    if(out_has_halo) {
	outs = m_stencil->get_delta_extent();
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
    convolve_with_gaussian();
    blur_rhs();
    
    // r = rho - Adot(phi,xsize,ysize,zsize)
    adot(m_phi, m_Ap, m_phi_haloex);
    axpby(1.0, m_blurred_rho, -1.0, m_Ap, m_r, false, false, false);

    // p = r  
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
	m_logger->trace("cg it %d: |r| = %g", i, sqrt(rr));
	m_logger->trace("cg it %d: |r|/|phi| = %g", i, err);

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
