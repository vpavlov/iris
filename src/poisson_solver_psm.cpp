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
#include "utils.h"

using namespace ORG_NCSA_IRIS;

static const iris_real _PI  =  3.141592653589793;
static const iris_real _2PI =  6.283185307179586;
static const iris_real _4PI = 12.566370614359172;

poisson_solver_psm::poisson_solver_psm(iris *obj)
    :poisson_solver(obj), m_ev(NULL), m_fft(NULL), m_work1(NULL), m_work2(NULL)
{
    m_logger->info("Will use the Pseudo-spectral method for solving Poisson's equation");
}

poisson_solver_psm::~poisson_solver_psm()
{
    if(m_fft != NULL) {
	delete m_fft;
    }

    memory::destroy_3d(m_ev);
    memory::destroy_1d(m_work1);
    memory::destroy_1d(m_work2);
}

void poisson_solver_psm::calculate_eigenvalues()
{
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];
    
    int sx = m_mesh->m_own_offset[0];
    int sy = m_mesh->m_own_offset[1];
    int sz = m_mesh->m_own_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    int norm = m_mesh->m_size[0] * m_mesh->m_size[1] * m_mesh->m_size[2];

    int hx = m_mesh->m_h[0];
    int hy = m_mesh->m_h[1];
    int hz = m_mesh->m_h[2];

    int cnt = m_laplacian->m_acc/2;
    int cnt1 = cnt+1;
    iris_real *A = (iris_real *)memory::wmalloc((cnt+1)*sizeof(iris_real));
    iris_real *B = (iris_real *)memory::wmalloc((cnt+1)*sizeof(iris_real));
    iris_real *C = (iris_real *)memory::wmalloc((cnt+1)*sizeof(iris_real));
    
    A[0] = B[0] = C[0] = 1.0;

    iris_real *data = (iris_real *)m_laplacian->m_data;

    memory::destroy_3d(m_ev);
    memory::create_3d(m_ev, nx, ny, nz);
    for(int x = sx; x < ex; x++) {
    	iris_real t = _2PI * x / nx;
    	for(int c = 1; c <= cnt; c++) {
    	    A[c] = 2*cos(c*t);
    	}
	
    	for(int y = sy; y < ey; y++) {
    	    iris_real t = _2PI * y / ny;
    	    for(int c = 1; c <= cnt; c++) {
    		B[c] = 2*cos(c*t);
    	    }
	    
    	    for(int z = sz; z < ez; z++) {
    		iris_real t = _2PI * z / nz;
    		for(int c = 1; c <= cnt; c++) {
    		    C[c] = 2*cos(c*t);
    		}
		
    		iris_real val = (iris_real)0.0;
    		for(int i=0;i<=cnt;i++) {
    		    int xl = cnt - i;
    		    for(int j=0;j<=cnt;j++) {
    			int yl = cnt - j;
    			for(int k=0;k<=cnt;k++) {
    			    int zl = cnt - k;
    			    iris_real sc = data[ROW_MAJOR_OFFSET(i, j, k, cnt1, cnt1)];
    			    iris_real tt = sc*A[xl]*B[yl]*C[zl];
    			    val += tt;
    			}
    		    }
    		}
		
    		if(val == 0.0) {
    		    val = std::numeric_limits<iris_real>::min();
    		}
		
    		m_ev[x-sx][y-sy][z-sz] = val * norm;
    	    }
    	}
    }
    
    m_logger->trace("Pseudo-spectral method: Laplacian eigenvalues calculated");
}

void poisson_solver_psm::commit()
{
    bool dirty = m_dirty;
    poisson_solver::commit();
    if(dirty) {
	m_laplacian->set_hx(m_mesh->m_h[0]);
	m_laplacian->set_hy(m_mesh->m_h[1]);
	m_laplacian->set_hz(m_mesh->m_h[2]);
	m_laplacian->commit();

	calculate_eigenvalues();

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

void poisson_solver_psm::divide_by_eigenvalues(iris_real *krho)
{
    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];
    
    int sx = m_mesh->m_own_offset[0];
    int sy = m_mesh->m_own_offset[1];
    int sz = m_mesh->m_own_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    int idx = 0;
    for(int i=0;i<nx;i++) {
	for(int j=0;j<ny;j++) {
	    for(int k=0;k<nz;k++) {
		krho[idx]   /= m_ev[i][j][k];
		krho[idx+1] /= m_ev[i][j][k];
		idx+=2;
	    }
	}
    }

    // make sure that the first coeff. is not infinity (e.g. not entirely
    // perfect neutral box leads to this).
    krho[0] = krho[1] = 0.0;
}

void poisson_solver_psm::solve()
{
    m_logger->trace("Solving Poisson's Equation now");
    m_fft->compute_fw(&(m_mesh->m_rho[0][0][0]), m_work1);
    divide_by_eigenvalues(m_work1);
    m_fft->compute_bk(m_work1, &(m_mesh->m_phi[0][0][0]));
}

void poisson_solver_psm::dump_work(int i)
{
    iris_real *data = NULL;
    if(i==1) {
	data = m_work1;
    }else if(i==2) {
	data = m_work2;
    }

    for(int i=0;i<m_fft->m_count;i++) {
	m_logger->trace("FFT[%d] = %.16f + j*%.16f",
			i, data[i*2+0], data[i*2+1]);
    }
}
