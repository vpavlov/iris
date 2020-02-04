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
#include "poisson_solver_p3m.h"
#include "domain.h"
#include "mesh.h"
#include "charge_assigner.h"
#include "memory.h"
#include "math_util.h"
#include "logger.h"
#include "fft3D.h"
#include "openmp.h"
#include "timer.h"
#include "remap.h"
#include "grid.h"

#define  _PI   3.141592653589793238462643383279
#define _2PI   6.283185307179586476925286766559
#define _4PI  12.56637061435917295385057353311
#define  EPS   1.0e-7

using namespace ORG_NCSA_IRIS;

poisson_solver_p3m::poisson_solver_p3m(class iris *obj)
    : poisson_solver(obj), m_greenfn(NULL), m_kx(NULL), m_ky(NULL), m_kz(NULL),
      m_fft1(NULL), m_fft2(NULL),
      m_work1(NULL), m_work2(NULL), m_work3(NULL),
      m_remap(NULL), m_fft_grid(NULL), m_fft_size { 0, 0, 0 }, m_fft_offset { 0, 0, 0 }
{
};

poisson_solver_p3m::~poisson_solver_p3m()
{
    memory::destroy_3d(m_greenfn);
    memory::destroy_1d(m_kx);
    memory::destroy_1d(m_ky);
    memory::destroy_1d(m_kz);
    memory::destroy_1d(m_work1);
    memory::destroy_1d(m_work2);
    memory::destroy_1d(m_work3);
    if(m_fft1) { delete m_fft1; }
    if(m_fft2) { delete m_fft2; }
    if(m_fft_grid) { delete m_fft_grid; }
    if(m_remap) { delete m_remap; }
}

void poisson_solver_p3m::commit()
{
    if(!m_dirty) {
	return;
    }

    if(m_fft_grid) { delete m_fft_grid; }

    m_fft_grid = new grid(m_iris, "P3M FFT GRID");
    m_fft_grid->set_pref(0, 1, 1);  // e.g. grid will be 64x1x1, mesh will be 2x128x128
    m_fft_grid->commit();

    m_fft_size[0] = m_mesh->m_size[0] / m_fft_grid->m_size[0];
    m_fft_size[1] = m_mesh->m_size[1] / m_fft_grid->m_size[1];
    m_fft_size[2] = m_mesh->m_size[2] / m_fft_grid->m_size[2];

    int *c = m_fft_grid->m_coords;
    m_fft_offset[0] = c[0] * m_fft_size[0];
    m_fft_offset[1] = c[1] * m_fft_size[1];
    m_fft_offset[2] = c[2] * m_fft_size[2];

    if(m_remap) { delete m_remap; }

    m_remap = new remap(m_iris,
			m_mesh->m_own_offset,
			m_mesh->m_own_size,
			m_fft_offset,
			m_fft_size,
			1,
			0, "initial_remap");

    memory::destroy_3d(m_greenfn);
    memory::create_3d(m_greenfn, m_fft_size[0], m_fft_size[1], m_fft_size[2]);

    memory::destroy_1d(m_kx);
    memory::create_1d(m_kx, m_fft_size[0]);

    memory::destroy_1d(m_ky);
    memory::create_1d(m_ky, m_fft_size[1]);

    memory::destroy_1d(m_kz);
    memory::create_1d(m_kz, m_fft_size[2]);

    calculate_green_function();
    calculate_k();

    if(m_fft1 != NULL) { delete m_fft1; }
    m_fft1 = new fft3d(m_iris,
		       m_fft_offset, m_fft_size,
		       m_fft_offset, m_fft_size, "fft1");
		       

    if(m_fft2 != NULL) { delete m_fft2; }
    m_fft2 = new fft3d(m_iris,
		       m_fft_offset, m_fft_size,
		       m_mesh->m_own_offset, m_mesh->m_own_size, "fft2");
    
    int n = 2 * m_fft1->m_count;
    
    memory::destroy_1d(m_work1);
    memory::create_1d(m_work1, n);
    
    memory::destroy_1d(m_work2);
    memory::create_1d(m_work2, n);

    memory::destroy_1d(m_work3);
    memory::create_1d(m_work3, n);
    
    m_dirty = false;
}

void poisson_solver_p3m::kspace_phi(iris_real *io_rho_phi)
{
    iris_real scaleinv = 1.0/(m_mesh->m_size[0] * m_mesh->m_size[1] * m_mesh->m_size[2]);

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		io_rho_phi[idx++] *= scaleinv * m_greenfn[i][j][k];
		io_rho_phi[idx++] *= scaleinv * m_greenfn[i][j][k];
	    }
	}
    }
}

void poisson_solver_p3m::kspace_eng(iris_real *in_rho_phi)
{
    // FFT is not normalized, so we need to do that now
    iris_real s2 = square(1.0/(m_mesh->m_size[0] * m_mesh->m_size[1] * m_mesh->m_size[2]));

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		m_iris->m_Ek += s2 * m_greenfn[i][j][k] *
		    (in_rho_phi[idx  ] * in_rho_phi[idx  ] +
		     in_rho_phi[idx+1] * in_rho_phi[idx+1]);
		idx += 2;
	    }
	}
    }
}

void poisson_solver_p3m::kspace_Ex(iris_real *in_phi, iris_real *out_Ex)
{
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		out_Ex[idx]   =  in_phi[idx+1]*m_kx[i];
		out_Ex[idx+1] = -in_phi[idx  ]*m_kx[i];
		idx+=2;
	    }
	}
    }
}

void poisson_solver_p3m::kspace_Ey(iris_real *in_phi, iris_real *out_Ey)
{
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		out_Ey[idx]   =  in_phi[idx+1]*m_ky[j];
		out_Ey[idx+1] = -in_phi[idx  ]*m_ky[j];
		idx+=2;
	    }
	}
    }
}

void poisson_solver_p3m::kspace_Ez(iris_real *in_phi, iris_real *out_Ez)
{
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		out_Ez[idx]   =  in_phi[idx+1]*m_kz[k];
		out_Ez[idx+1] = -in_phi[idx  ]*m_kz[k];
		idx+=2;
	    }
	}
    }
}

// Hockney/Eastwood modified Green function corresponding to the
// charge assignment functions
//
// G(k) = _4PI/k^2 * [ sum_b(k.(k+b)/(k+b).(k+b)  *  Wn^2(k+b) * rho^2(k+b) ] / sum_b(Wn^2(k+b))^2
//
void poisson_solver_p3m::calculate_green_function()
{
    const iris_real alpha = m_iris->m_alpha;

    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];

    const int nbx = static_cast<int> ((alpha*xL/(_PI*xM)) * pow(-log(EPS),0.25));
    const int nby = static_cast<int> ((alpha*yL/(_PI*yM)) * pow(-log(EPS),0.25));
    const int nbz = static_cast<int> ((alpha*zL/(_PI*zM)) * pow(-log(EPS),0.25));

    const int _2n = 2*m_chass->m_order;

#if defined _OPENMP
#pragma omp parallel default(none)
#endif

    {
	int nx = m_fft_size[0];
	int ny = m_fft_size[1];
	int nz = m_fft_size[2];
	
	int sx = m_fft_offset[0];
	int sy = m_fft_offset[1];
	int sz = m_fft_offset[2];
	
	int ex = sx + nx;
	int ey = sy + ny;
	int ez = sz + nz;
	
	int from, to;
	setup_work_sharing(nx, m_iris->m_nthreads, &from, &to);

	// // k = 2pij/L
	// // h = L/M
	// // kh/2 = 2pij/L * L/M = 2pi*j/M
	int n = 0;
	for(int x = sx + from; x < sx + to; x++) {
	    int xj = x - xM*(2*x/xM);
	    iris_real sinx2 = square(sin(_PI*xj/xM));
	    
	    for(int y = sy; y < ey; y++) {
		int yj = y - yM*(2*y/yM);
		iris_real siny2 = square(sin(_PI*yj/yM));
		
		for(int z = sz; z < ez; z++) {
		    int zj = z - zM*(2*z/zM);  // convert from 0..P to 0..P/2, -P/2...-1
		    iris_real sinz2 = square(sin(_PI*zj/zM));  // sin^2(k*delta/2)
		    
		    iris_real ksq = square(kxm * xj) + square(kym * yj) + square(kzm * zj);
		    
		    if(ksq != 0.0) {
			iris_real part1 = _4PI / ksq;
			iris_real part2 = 0.0;
			
			for(int bx = -nbx; bx <= nbx; bx++) {
			    iris_real xkplusb = kxm * (xj + xM*bx);
			    iris_real xrho = exp(-0.25*square(xkplusb/alpha));
			    iris_real xwnsq = pow_sinx_x(xkplusb*xL/(2*xM), _2n);
			    
			    for(int by = -nby; by <= nby; by++) {
				iris_real ykplusb = kym * (yj + yM*by);
				iris_real yrho = exp(-0.25*square(ykplusb/alpha));
				iris_real ywnsq = pow_sinx_x(ykplusb*yL/(2*yM), _2n);
				
				for(int bz = -nbz; bz <= nbz; bz++) {
				    iris_real zkplusb = kzm * (zj + zM*bz);
				    iris_real zrho = exp(-0.25*square(zkplusb/alpha));
				    iris_real zwnsq = pow_sinx_x(zkplusb*zL/(2*zM), _2n);
				    
				    // k . (k+b)
				    iris_real k_dot_kplusb =
					kxm * xj * xkplusb +
					kym * yj * ykplusb +
					kzm * zj * zkplusb;
				    
				    // (k+b) . (k+b)
				    iris_real kplusb_sq = xkplusb * xkplusb + ykplusb*ykplusb + zkplusb*zkplusb;
				    
				    part2 += (k_dot_kplusb/kplusb_sq) * xrho*yrho*zrho * xwnsq * ywnsq * zwnsq;
				}
			    }
			}
			iris_real part3 = denominator(sinx2, siny2, sinz2);

			m_greenfn[x-sx][y-sy][z-sz] = part1 * part2 / part3;
		    }else {
			m_greenfn[x-sx][y-sy][z-sz] = 0.0;
		    }
		}
	    }
	}
    }
}

void poisson_solver_p3m::calculate_k()
{
    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;
    
    // k = 2pij/L
    // h = L/M
    // kh/2 = 2pij/L * L/M = 2pi*j/M

    for(int x = sx; x < ex; x++) {
	int xj = x - xM*(2*x/xM);
	m_kx[x-sx] = kxm * xj;
    }

    for(int y = sy; y < ey; y++) {
	int yj = y - yM*(2*y/yM);
	m_ky[y-sy] = kym * yj;
    }

    for(int z = sz; z < ez; z++) {
	int zj = z - zM*(2*z/zM);
	m_kz[z-sz] = kzm * zj;
    }
}

void poisson_solver_p3m::solve()
{
    m_logger->trace("Solving Poisson's Equation now");

    m_remap->perform(&(m_mesh->m_rho[0][0][0]), m_work2, m_work1);
    m_fft1->compute_fw(m_work2, m_work1);

    if(m_iris->m_compute_global_energy) {
	kspace_eng(m_work1);
    }

    kspace_phi(m_work1);

    kspace_Ex(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, &(m_mesh->m_Ex[0][0][0]));
    
    kspace_Ey(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, &(m_mesh->m_Ey[0][0][0]));
    
    kspace_Ez(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, &(m_mesh->m_Ez[0][0][0]));

    m_fft2->compute_bk(m_work1, &(m_mesh->m_phi[0][0][0]));
}
