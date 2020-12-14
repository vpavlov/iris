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
#include "comm_rec.h"
#include "utils.h"

#define  _PI   3.141592653589793238462643383279
#define _2PI   6.283185307179586476925286766559
#define _4PI  12.56637061435917295385057353311
#define  EPS   1.0e-7

using namespace ORG_NCSA_IRIS;

poisson_solver_p3m::poisson_solver_p3m(class iris *obj)
    : solver(obj), m_greenfn(NULL), 
      m_denominator_x(NULL), m_denominator_y(NULL), m_denominator_z(NULL), 
      m_kx(NULL), m_ky(NULL), m_kz(NULL), m_vc(NULL),
      m_fft1(NULL), m_fft2(NULL),
      m_work1(NULL), m_work2(NULL), m_work3(NULL),
      m_remap(NULL), m_fft_grid(NULL), m_fft_size { 0, 0, 0 }, m_fft_offset { 0, 0, 0 }
{
};

poisson_solver_p3m::~poisson_solver_p3m()
{
    memory::destroy_1d(m_greenfn);
    memory::destroy_1d(m_denominator_x);    
    memory::destroy_1d(m_denominator_y);    
    memory::destroy_1d(m_denominator_z);
    memory::destroy_1d(m_kx);
    memory::destroy_1d(m_ky);
    memory::destroy_1d(m_kz);
    memory::destroy_2d(m_vc);
    memory::destroy_1d(m_work1);
    memory::destroy_1d(m_work2);
    memory::destroy_1d(m_work3);
    if(m_fft1) { delete m_fft1; }
    if(m_fft2) { delete m_fft2; }
    if(m_fft_grid) { delete m_fft_grid; }
    if(m_remap) { delete m_remap; }
}

void poisson_solver_p3m::handle_box_resize()
{
    calculate_green_function();
    calculate_k();
    calculate_virial_coeff();
}

void poisson_solver_p3m::commit()
{
    if(!m_dirty) {
	return;
    }

    solver_param_t p = m_iris->get_solver_param(IRIS_SOLVER_P3M_USE_COLLECTIVE);
    bool use_collective = (p.i == 1)?true:false;
    
    if(m_fft_grid) { delete m_fft_grid; }

    m_fft_grid = new grid(m_iris, "P3M FFT GRID");
    if (m_mesh->m_size[0] > m_local_comm->m_size) {
	m_fft_grid->set_pref(0, 1, 1);  // e.g. grid will be 64x1x1, mesh will be 2x128x128
    }else if (m_mesh->m_size[0]*m_mesh->m_size[1] > m_local_comm->m_size) {
	m_fft_grid->set_pref(0, 0, 1);  // e.g. grid will be 64x2x1, mesh will be 2x64x128
    }

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
			0, "initial_remap",
			use_collective);

    memory::destroy_1d(m_greenfn);
    memory::create_1d(m_greenfn, m_fft_size[0] * m_fft_size[1] * m_fft_size[2]);

    memory::destroy_1d(m_kx);
    memory::create_1d(m_kx, m_fft_size[0]);

    memory::destroy_1d(m_ky);
    memory::create_1d(m_ky, m_fft_size[1]);

    memory::destroy_1d(m_kz);
    memory::create_1d(m_kz, m_fft_size[2]);

    memory::destroy_2d(m_vc);
    memory::create_2d(m_vc, m_fft_size[0]*m_fft_size[1]*m_fft_size[2], 6);

    if (m_denominator_x==NULL) {
	memory::create_1d(m_denominator_x, m_fft_size[0]);
	memory::create_1d(m_denominator_y, m_fft_size[1]);
	memory::create_1d(m_denominator_z, m_fft_size[2]);
	calculate_denominator();
    }
    calculate_green_function();
    calculate_k();
    calculate_virial_coeff();

    if(m_fft1 != NULL) { delete m_fft1; }
    m_fft1 = new fft3d(m_iris,
		       m_fft_offset, m_fft_size,
		       m_fft_offset, m_fft_size, "fft1", use_collective);
		       

    if(m_fft2 != NULL) { delete m_fft2; }
    m_fft2 = new fft3d(m_iris,
		       m_fft_offset, m_fft_size,
		       m_mesh->m_own_offset, m_mesh->m_own_size, "fft2", use_collective);
    
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
    int n = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		io_rho_phi[idx++] *= scaleinv * m_greenfn[n];
		io_rho_phi[idx++] *= scaleinv * m_greenfn[n];
		n++;
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

    if(m_iris->m_compute_global_virial) {
	int idx = 0;
	int n = 0;
	for(int i=0;i<nx;i++) {
	    for(int j=0;j<ny;j++) {
		for(int k=0;k<nz;k++) {
		    iris_real ener = s2 * m_greenfn[n++] *
			(in_rho_phi[idx  ] * in_rho_phi[idx  ] +
			 in_rho_phi[idx+1] * in_rho_phi[idx+1]);
		    for(int m = 0;m<6;m++) {
			m_iris->m_virial[m] += ener * m_vc[idx/2][m];
		    }
		    if(m_iris->m_compute_global_energy) {
			m_iris->m_Ek += ener;
		    }
		    idx += 2;
		}
	    }
	}
    }else {
	int idx = 0;
	int n = 0;
	for(int i=0;i<nx;i++) {
	    for(int j=0;j<ny;j++) {
		for(int k=0;k<nz;k++) {
		    m_iris->m_Ek += s2 * m_greenfn[n++] *
			(in_rho_phi[idx  ] * in_rho_phi[idx  ] +
			 in_rho_phi[idx+1] * in_rho_phi[idx+1]);
		    idx += 2;
		}
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

void poisson_solver_p3m::calculate_denominator()
{

#if defined _OPENMP
#pragma omp parallel default(none)
#endif
    {

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
	    
	int from, to;
	    
	setup_work_sharing(nx, m_iris->m_nthreads, &from, &to);
	for (int x = sx + from; x < sx + to; x++) {
	    int xj = x - xM * (2 * x / xM);
	    iris_real sinx2 = square(sin(_PI * xj / xM));
	    m_denominator_x[x - sx] = denominator1(sinx2);
	}

	setup_work_sharing(ny, m_iris->m_nthreads, &from, &to);
	for (int y = sy + from; y < sy + to; y++) {
	    int yj = y - yM * (2 * y / yM);
	    iris_real siny2 = square(sin(_PI * yj / yM));
	    m_denominator_y[y - sy] = denominator1(siny2);
	}

	setup_work_sharing(nz, m_iris->m_nthreads, &from, &to);
	for (int z = sz + from; z < sz + to; z++) {
	    int zj = z - zM * (2 * z / zM);
	    iris_real sinz2 = square(sin(_PI * zj / zM));
	    m_denominator_z[z - sz] = denominator1(sinz2);
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

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];
    
    const int nbx = static_cast<int> ((alpha*xL/(_PI*xM)) * pow(-log(EPS),0.25));
    const int nby = static_cast<int> ((alpha*yL/(_PI*yM)) * pow(-log(EPS),0.25));
    const int nbz = static_cast<int> ((alpha*zL/(_PI*zM)) * pow(-log(EPS),0.25));

    if(nbx == 0 && nby == 0 && nbz == 0) {
	calculate_gf_fact();
    }else {
	calculate_gf_full();
    }
}

void poisson_solver_p3m::calculate_gf_fact()
{
    const iris_real alpha = m_iris->m_alpha;

    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];
    
    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);
	
    const int _2n = 2*m_chass->m_order;

    iris_real *greenfn_x;
    iris_real *greenfn_y;
    iris_real *greenfn_z;
	
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    memory::create_1d(greenfn_x, nx);
    memory::create_1d(greenfn_y, ny);
    memory::create_1d(greenfn_z, nz);
    
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int from, to;
	setup_work_sharing(nx, m_iris->m_nthreads, &from, &to);
	
	// k = 2pij/L
	// h = L/M
	// kh/2 = 2pij/L * L/M = 2pi*j/M
	for (int x = sx + from; x < sx + to; x++) {
	    int xj = x - xM * (2 * x / xM);
	    iris_real xkplusb = kxm * xj;
	    iris_real xrho = exp(-0.25 * square(xkplusb / alpha));
	    iris_real xwnsq = pow_sinx_x(xkplusb * xL / (2 * xM), _2n);
	    iris_real part2 = xrho * xwnsq;
	    greenfn_x[x - sx] = part2 / m_denominator_x[x - sx];
	}
    }


#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int from, to;
	setup_work_sharing(ny, m_iris->m_nthreads, &from, &to);
	for (int y = sy + from; y < sy + to; y++) {
	    int yj = y - yM * (2 * y / yM);
	    iris_real ykplusb = kym * yj;
	    iris_real yrho = exp(-0.25 * square(ykplusb / alpha));
	    iris_real ywnsq = pow_sinx_x(ykplusb * yL / (2 * yM), _2n);
	    iris_real part2 = yrho * ywnsq;
	    greenfn_y[y - sy] = part2 / m_denominator_y[y - sy];
	}
    }
	    

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int from, to;
	setup_work_sharing(nz, m_iris->m_nthreads, &from, &to);
	for (int z = sz + from; z < sz + to; z++) {
	    int zj = z - zM * (2 * z / zM);
	    
	    iris_real zkplusb = kzm * zj;
	    iris_real zrho = exp(-0.25 * square(zkplusb / alpha));
	    iris_real zwnsq = pow_sinx_x(zkplusb * zL / (2 * zM), _2n);
	    
	    iris_real part2 = zrho * zwnsq;
	    
	    greenfn_z[z - sz] = part2 / m_denominator_z[z - sz];
	}
    }
	    

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int from, to;
	setup_work_sharing(nx, m_iris->m_nthreads, &from, &to);
	for (int x = sx + from; x < sx + to; x++) {
	    int xj = x - xM * (2 * x / xM);
	    for (int y = sy; y < ey; y++) {
		int yj = y - yM * (2 * y / yM);
		for (int z = sz; z < ez; z++) {
		    int zj = z - zM * (2 * z / zM); // convert from 0..P to 0..P/2, -P/2...-1
		    iris_real ksq = square(kxm * xj) + square(kym * yj) + square(kzm * zj);
		    if (ksq != 0.0) {
			iris_real part1 = _4PI / ksq;
			iris_real part2 = greenfn_x[x - sx] * greenfn_y[y - sy] * greenfn_z[z - sz];
			m_greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = part1 * part2;
		    }else {
			m_greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = 0.0;
		    }
		}
	    }
	}
    }
	
    memory::destroy_1d(greenfn_x);
    memory::destroy_1d(greenfn_y);
    memory::destroy_1d(greenfn_z);
}

void poisson_solver_p3m::calculate_gf_full()
{
    const iris_real alpha = m_iris->m_alpha;

    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];

    const int nbx = static_cast<int> ((alpha*xL/(_PI*xM)) * pow(-log(EPS),0.25));
    const int nby = static_cast<int> ((alpha*yL/(_PI*yM)) * pow(-log(EPS),0.25));
    const int nbz = static_cast<int> ((alpha*zL/(_PI*zM)) * pow(-log(EPS),0.25));
    
    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);
	
    const int _2n = 2*m_chass->m_order;

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int from, to;
	setup_work_sharing(nx, m_iris->m_nthreads, &from, &to);
	    
	// k = 2pij/L
	// h = L/M
	// kh/2 = 2pij/L * L/M = 2pi*j/M
	int n = 0;
	for (int x = sx + from; x < sx + to; x++) {
	    int xj = x - xM * (2 * x / xM);
	    for (int y = sy; y < ey; y++) {
		int yj = y - yM * (2 * y / yM);
		for (int z = sz; z < ez; z++) {
		    int zj = z - zM * (2 * z / zM);               // convert from 0..P to 0..P/2, -P/2...-1
		    iris_real ksq = square(kxm * xj) + square(kym * yj) + square(kzm * zj);
		    if (ksq != 0.0) {
			iris_real part1 = _4PI / ksq;
			iris_real part2 = 0.0;
			for (int bx = -nbx; bx <= nbx; bx++) {
			    iris_real xkplusb = kxm * (xj + xM * bx);
			    iris_real xrho = exp(-0.25 * square(xkplusb / alpha));
			    iris_real xwnsq = pow_sinx_x(xkplusb * xL / (2 * xM), _2n);

			    for (int by = -nby; by <= nby; by++) {
				iris_real ykplusb = kym * (yj + yM * by);
				iris_real yrho = exp(-0.25 * square(ykplusb / alpha));
				iris_real ywnsq = pow_sinx_x(ykplusb * yL / (2 * yM), _2n);
							    
				for (int bz = -nbz; bz <= nbz; bz++) {
				    iris_real zkplusb = kzm * (zj + zM * bz);
				    iris_real zrho = exp(-0.25 * square(zkplusb / alpha));
				    iris_real zwnsq = pow_sinx_x(zkplusb * zL / (2 * zM), _2n);
				    
				    // k . (k+b)
				    iris_real k_dot_kplusb = kxm * xj * xkplusb + kym * yj * ykplusb + kzm * zj * zkplusb;
								    
				    // (k+b) . (k+b)
				    iris_real kplusb_sq = xkplusb * xkplusb + ykplusb * ykplusb + zkplusb * zkplusb;
								    
				    part2 += (k_dot_kplusb / kplusb_sq) * xrho * yrho * zrho * xwnsq * ywnsq * zwnsq;
				}
			    }
			}
			iris_real part3 = m_denominator_x[x - sx]*m_denominator_y[y - sy]*m_denominator_z[z - sz];
			m_greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = part1 * part2 / part3;
		    }else {
			m_greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = 0.0;
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

    // k' = 2πn'/L, where n' = n for 0 <= n < N/2 and = n-N otherwise
    // As descried in 9807099.pdf towards the bottom of page 6
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

void poisson_solver_p3m::calculate_virial_coeff()
{
    const iris_real alpha = m_iris->m_alpha;

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    int n = 0;
    for(int x = sx; x < ex; x++) {
	for(int y = sy; y < ey; y++) {
	    for(int z = sz; z < ez; z++) {
		iris_real sq =
		    m_kx[x-sx]*m_kx[x-sx] +
		    m_ky[y-sy]*m_ky[y-sy] +
		    m_kz[z-sz]*m_kz[z-sz];
		if(sq == 0.0) {
		    m_vc[n][0] = m_vc[n][1] = m_vc[n][2] =
			m_vc[n][3] = m_vc[n][4] = m_vc[n][5] = 0.0;
		}else {
		    iris_real t = -2.0/sq - 0.5/(alpha * alpha);
		    m_vc[n][0] = 1.0 + t * m_kx[x-sx] * m_kx[x-sx];
		    m_vc[n][1] = 1.0 + t * m_ky[y-sy] * m_ky[y-sy];
		    m_vc[n][2] = 1.0 + t * m_kz[z-sz] * m_kz[z-sz];
		    m_vc[n][3] = t * m_kx[x-sx] * m_ky[y-sy];
		    m_vc[n][4] = t * m_kx[x-sx] * m_kz[z-sz];
		    m_vc[n][5] = t * m_ky[y-sy] * m_kz[z-sz];
		}
		n++;
	    }
	}
    }
}

void poisson_solver_p3m::solve()
{
    m_logger->trace("P3M solve() start");

    if(m_iris->m_compute_global_energy) {
	m_iris->m_Ek = 0.0;
    }

    if(m_iris->m_compute_global_virial) {
	m_iris->m_virial[0] =
	    m_iris->m_virial[1] =
	    m_iris->m_virial[2] =
	    m_iris->m_virial[3] =
	    m_iris->m_virial[4] =
	    m_iris->m_virial[5] = 0.0;
    }
    
    m_remap->perform(&(m_mesh->m_rho[0][0][0]), m_work2, m_work1);
    m_fft1->compute_fw(m_work2, m_work1);

    if(m_iris->m_compute_global_energy || m_iris->m_compute_global_virial) {
	kspace_eng(m_work1);
    }

    kspace_phi(m_work1);

    kspace_Ex(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, &(m_mesh->m_Ex[0][0][0]));
    
    kspace_Ey(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, &(m_mesh->m_Ey[0][0][0]));
    
    kspace_Ez(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, &(m_mesh->m_Ez[0][0][0]));

    //////////////// we do not need this ////////////////////////
    // m_fft2->compute_bk(m_work1, &(m_mesh->m_phi[0][0][0])); //
    /////////////////////////////////////////////////////////////
    
    iris_real post_corr = 0.5 *
	m_domain->m_global_box.xsize *
	m_domain->m_global_box.ysize *
	m_domain->m_global_box.zsize *
	m_units->ecf;
    
    if(m_iris->m_compute_global_energy) {
	m_iris->m_Ek *= post_corr;
    }
    
    if(m_iris->m_compute_global_virial) {
	for(int i=0;i<6;i++) {
	    m_iris->m_virial[i] *= post_corr;
	}
    }
}
