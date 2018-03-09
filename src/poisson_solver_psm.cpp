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
#include <cmath>
#include <limits>
#include <stdexcept>
#include "poisson_solver_psm.h"
#include "logger.h"
#include "mesh.h"
#include "memory.h"
#include "stencil.h"
#include "grid.h"
#include "comm_rec.h"
#include "remap.h"
#include "fft3D.h"
#include "domain.h"
#include "laplacian3D.h"
#include "first_derivative.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

static const iris_real _PI  =  3.141592653589793;
static const iris_real _2PI =  6.283185307179586;

poisson_solver_psm::poisson_solver_psm(iris *obj)
    :poisson_solver(obj), m_ev(NULL), m_ddx_ev(NULL),
     m_ddy_ev(NULL), m_ddz_ev(NULL), m_fft(NULL), m_work1(NULL), m_work2(NULL)
{
    m_logger->info("Will use the Pseudo-spectral method for solving Poisson's equation");
}

poisson_solver_psm::~poisson_solver_psm()
{
    if(m_fft != NULL) {
	delete m_fft;
    }

    memory::destroy_3d(m_ev);
    memory::destroy_1d(m_ddx_ev);
    memory::destroy_1d(m_ddy_ev);
    memory::destroy_1d(m_ddz_ev);
    memory::destroy_1d(m_work1);
    memory::destroy_1d(m_work2);
}

void poisson_solver_psm::calculate_ddx_ev(int idx, first_derivative *ddx, 
					  iris_real *&ddx_ev)
{
    int gnx = m_mesh->m_size[idx];       // global mesh size
    int nx = m_mesh->m_own_size[idx];    // local mesh size;
    int sx = m_mesh->m_own_offset[idx];  // from where we start
    int ex = sx + nx;                  // where we end

    int cnt = ddx->m_acc;
    int cnt1 = cnt + 1;
    iris_real *A = (iris_real *)memory::wmalloc(cnt1*sizeof(iris_real));

    A[0] = 0.0;

    iris_real *data = (iris_real *)ddx->m_delta;

    memory::destroy_1d(ddx_ev);
    memory::create_1d(ddx_ev, nx);
    for(int x = sx; x < ex; x++) {
	iris_real t = _2PI * x / gnx;
	for(int c = 1; c < cnt1; c++) {
	    A[c] = 2*sin(c*t);
	}

	iris_real val = (iris_real)0.0;
	for(int i=0;i<cnt;i++) {
	    int xl = cnt - i;
	    iris_real sc = data[i];
	    iris_real tt = sc*A[xl];
	    val += tt;
	}

	ddx_ev[x-sx] = val;
    }
}

void poisson_solver_psm::calculate_laplacian_ev()
{
    // global mesh size
    int gnx = m_mesh->m_size[0];
    int gny = m_mesh->m_size[1];
    int gnz = m_mesh->m_size[2];

    // local mesh size
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];

    // from where we start
    int sx = m_mesh->m_own_offset[0];
    int sy = m_mesh->m_own_offset[1];
    int sz = m_mesh->m_own_offset[2];

    // where we end
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    // normalization coefficient (FFT is not normalized)
    int norm = gnx * gny * gnz;

    int cnt_d = m_laplacian->get_delta_extent();
    int cnt1_d = cnt_d+1;  // + the center point

    int cnt_g = m_laplacian->get_gamma_extent();
    int cnt1_g = cnt_g+1;  // + the center point

    int cnt1 = MAX(cnt1_d, cnt1_g);

    iris_real *A = (iris_real *)memory::wmalloc(cnt1*sizeof(iris_real));
    iris_real *B = (iris_real *)memory::wmalloc(cnt1*sizeof(iris_real));
    iris_real *C = (iris_real *)memory::wmalloc(cnt1*sizeof(iris_real));

    // A[0] is what we multiply the center coeff with
    // A[1] is what we multiply the coeff for 1-st additional layer with
    // ...
    // A[i] is what we multiply the coeff for the i-th additional layer with
    // Same for B and C
    A[0] = B[0] = C[0] = 1.0;

    memory::destroy_3d(m_ev);
    memory::create_3d(m_ev, nx, ny, nz);
    for(int x = sx; x < ex; x++) {
    	iris_real tx = _2PI * x / gnx;
    	for(int c = 1; c < cnt1; c++) {
    	    A[c] = 2*cos(c*tx);
    	}
	
    	for(int y = sy; y < ey; y++) {
    	    iris_real ty = _2PI * y / gny;
    	    for(int c = 1; c < cnt1; c++) {
    		B[c] = 2*cos(c*ty);
    	    }
	    
    	    for(int z = sz; z < ez; z++) {
    		iris_real tz = _2PI * z / gnz;
    		for(int c = 1; c < cnt1; c++) {
    		    C[c] = 2*cos(c*tz);
    		}
		
    		iris_real val_d = (iris_real)0.0;
    		for(int i=0;i<=cnt_d;i++) {
    		    int xl = cnt_d - i;
    		    for(int j=0;j<=cnt_d;j++) {
    			int yl = cnt_d - j;
    			for(int k=0;k<=cnt_d;k++) {
    			    int zl = cnt_d - k;
			    iris_real sc = m_laplacian->get_delta(i, j, k);
    			    iris_real tt = sc*A[xl]*B[yl]*C[zl];
    			    val_d += tt;
    			}
    		    }
    		}

    		if(val_d == 0.0) {
    		    val_d = std::numeric_limits<iris_real>::min();
    		}

    		iris_real val_g = (iris_real)0.0;
    		for(int i=0;i<=cnt_g;i++) {
    		    int xl = cnt_g - i;
    		    for(int j=0;j<=cnt_g;j++) {
    			int yl = cnt_g - j;
    			for(int k=0;k<=cnt_g;k++) {
    			    int zl = cnt_g - k;
			    iris_real sc = m_laplacian->get_gamma(i, j, k);
    			    iris_real tt = sc*A[xl]*B[yl]*C[zl];
    			    val_g += tt;
    			}
    		    }
    		}

    		if(val_g == 0.0) {
    		    val_g = std::numeric_limits<iris_real>::min();
    		}
		
		iris_real val = val_d / val_g;

    		m_ev[x-sx][y-sy][z-sz] = val * norm;
    	    }
    	}
    }

    // for(int i=0;i<nx;i++) {
    // 	for(int j=0;j<ny;j++) {
    // 	    for(int k=0;k<nz;k++) {
    // 		m_logger->trace("EV[%d][%d][%d] = %.15g", i, j, k, m_ev[i][j][k]);
    // 	    }
    // 	}
    // }
    m_logger->trace("Pseudo-spectral method: Laplacian eigenvalues calculated");
}

void poisson_solver_psm::commit()
{
    bool dirty = m_dirty;
    poisson_solver::commit();
    if(dirty) {
	// setup the laplacian and its eigenvalues
	m_laplacian->set_hx(m_mesh->m_h[0]);
	m_laplacian->set_hy(m_mesh->m_h[1]);
	m_laplacian->set_hz(m_mesh->m_h[2]);
	m_laplacian->commit();

	calculate_laplacian_ev();

	// setup the first derivatives in each direction, and their
	// eigenvalues
	m_ddx->set_h(m_mesh->m_h[0]);
	m_ddx->commit();
	calculate_ddx_ev(0, m_ddx, m_ddx_ev);

	m_ddy->set_h(m_mesh->m_h[1]);
	m_ddy->commit();
	calculate_ddx_ev(1, m_ddy, m_ddy_ev);

	m_ddz->set_h(m_mesh->m_h[2]);
	m_ddz->commit();
	calculate_ddx_ev(2, m_ddz, m_ddz_ev);

	if(m_fft != NULL) { delete m_fft; }
	m_fft = new fft3d(m_iris);
	
	int n = 2 * m_fft->m_count;

	memory::destroy_1d(m_work1);
	memory::create_1d(m_work1, n);

	memory::destroy_1d(m_work2);
	memory::create_1d(m_work2, n);

	m_dirty = false;
    }
}

void poisson_solver_psm::kspace_phi(iris_real *io_rho_phi)
{
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];
    
    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		io_rho_phi[idx]   /= m_ev[i][j][k];
		io_rho_phi[idx+1] /= m_ev[i][j][k];
		idx+=2;
	    }
	}
    }

    // make sure that the very first coeff. (DC) is not infinity
    // (e.g. not entirely perfect neutral box leads to this).
    int sx = m_mesh->m_own_offset[0];
    int sy = m_mesh->m_own_offset[1];
    int sz = m_mesh->m_own_offset[2];
    if(sx == 0 && sy == 0 && sz == 0) {
	io_rho_phi[0] = io_rho_phi[1] = 0.0;
    }
}

void poisson_solver_psm::kspace_Ex(iris_real *in_phi, iris_real *out_Ex)
{
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		out_Ex[idx]   =  in_phi[idx+1]*m_ddx_ev[i];
		out_Ex[idx+1] = -in_phi[idx  ]*m_ddx_ev[i];
		idx+=2;
	    }
	}
    }
}

void poisson_solver_psm::kspace_Ey(iris_real *in_phi, iris_real *out_Ey)
{
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		out_Ey[idx]   =  in_phi[idx+1]*m_ddy_ev[j];
		out_Ey[idx+1] = -in_phi[idx  ]*m_ddy_ev[j];
		idx+=2;
	    }
	}
    }
}

void poisson_solver_psm::kspace_Ez(iris_real *in_phi, iris_real *out_Ez)
{
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		out_Ez[idx]   =  in_phi[idx+1]*m_ddz_ev[k];
		out_Ez[idx+1] = -in_phi[idx  ]*m_ddz_ev[k];
		idx+=2;
	    }
	}
    }
}

void poisson_solver_psm::solve()
{
    m_logger->trace("Solving Poisson's Equation now");

    m_fft->compute_fw(&(m_mesh->m_rho[0][0][0]), m_work1);
    kspace_phi(m_work1);

    kspace_Ex(m_work1, m_work2);
    m_fft->compute_bk(m_work2, &(m_mesh->m_Ex[0][0][0]));

    kspace_Ey(m_work1, m_work2);
    m_fft->compute_bk(m_work2, &(m_mesh->m_Ey[0][0][0]));

    kspace_Ez(m_work1, m_work2);
    m_fft->compute_bk(m_work2, &(m_mesh->m_Ez[0][0][0]));

    m_fft->compute_bk(m_work1, &(m_mesh->m_phi[0][0][0]));
}

void poisson_solver_psm::dump_work(int idx)
{
    iris_real *data = NULL;
    if(idx==1) {
	data = m_work1;
    }else if(idx==2) {
	data = m_work2;
    }

    for(int i=0;i<m_fft->m_count;i++) {
	m_logger->trace("WORK%d[%d] = %.15g + j*%.15g",
			idx, i, data[i*2+0], data[i*2+1]);
    }
}
