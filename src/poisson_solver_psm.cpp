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
#include "fft_grid.h"
#include "comm_rec.h"
#include "remap.h"

using namespace ORG_NCSA_IRIS;

poisson_solver_psm::poisson_solver_psm(iris *obj)
    :poisson_solver(obj), m_ev(NULL), m_remap(NULL), m_rho(NULL),
     m_scratch(NULL)
{
    m_fft_grid = new fft_grid(obj);
    m_logger->info("Will use the Pseudo-spectral method for solving Poisson's equation");
}

poisson_solver_psm::~poisson_solver_psm()
{
    if(m_remap != NULL) {
	delete m_remap;
    }
    delete m_fft_grid;
    memory::destroy_3d(m_ev);
    memory::destroy_1d(m_rho);
    memory::destroy_1d(m_scratch);
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
    
    int cnt = (m_stencil->m_size_1d - 1) / 2;
    iris_real *A = (iris_real *)memory::wmalloc((cnt+1)*sizeof(iris_real));
    iris_real *B = (iris_real *)memory::wmalloc((cnt+1)*sizeof(iris_real));
    iris_real *C = (iris_real *)memory::wmalloc((cnt+1)*sizeof(iris_real));
    
    A[0] = B[0] = C[0] = 1.0;
    
    memory::destroy_3d(m_ev);
    memory::create_3d(m_ev, nx, ny, nz);
    for(int x = sx; x < ex; x++) {
	iris_real t = M_PI * x / nx;
	for(int c = 1; c <= cnt; c++) {
	    A[c] = 2*cos(2*c*t);
	}
	
	for(int y = sy; y < ey; y++) {
	    iris_real t = M_PI * y / ny;
	    for(int c = 1; c <= cnt; c++) {
		B[c] = 2*cos(2*c*t);
	    }
	    
	    for(int z = sz; z < ez; z++) {
		iris_real t = M_PI * z / nz;
		for(int c = 1; c <= cnt; c++) {
		    C[c] = 2*cos(2*c*t);
		}
		
		iris_real val = (iris_real)0.0;
		for(int i=0;i<=cnt;i++) {
		    int xl = cnt - i;
		    for(int j=0;j<=cnt;j++) {
			int yl = cnt - j;
			for(int k=0;k<=cnt;k++) {
			    int zl = cnt - k;
			    iris_real sc = m_stencil->m_coeff[i][j][k];
			    iris_real tt = sc*A[xl]*B[yl]*C[zl];
			    val += tt;
			}
		    }
		}
		
		if(val == 0.0) {
		    val = std::numeric_limits<iris_real>::min();
		}
		
		m_ev[x-sx][y-sy][z-sz] = val;
	    }
	}
    }
    
    m_logger->trace("Pseudo-spectral method: Laplacian eigenvalues calculated");
}

void poisson_solver_psm::setup_fft_grid()
{
    // To make all 1D FFTs local, we need to make sure that a proc owns
    // an entire dimension (say X), hence the first 1 in the calls to
    // set_user_pref below.
    //
    // If we can fit a whole XY (or XZ for that matter) plane, or several
    // such planes, this is preferrable, because the FFTs in the other
    // direction will also be local (there will be a local remap).
    //
    // Otherwise, just do a 2D decomposition, fixing X to 1. This leads to
    // each proc owning a set of "pencils".
    if(m_mesh->m_size[2] >= m_local_comm->m_size) {
	m_fft_grid->set_pref(1, 1, 0);
    }else {
	// If there are more processors than mesh size in z, then make each
	// processor own a 2D sub-blocks of the YZ-plane
	m_fft_grid->set_pref(1, 0, 0);
    }
    
    m_fft_grid->commit();

    m_own_size[0] = m_mesh->m_size[0] / m_fft_grid->m_size[0];
    m_own_size[1] = m_mesh->m_size[1] / m_fft_grid->m_size[1];
    m_own_size[2] = m_mesh->m_size[2] / m_fft_grid->m_size[2];

    int *c = m_fft_grid->m_coords;
    m_own_offset[0] = c[0] * m_own_size[0];
    m_own_offset[1] = c[1] * m_own_size[1];
    m_own_offset[2] = c[2] * m_own_size[2];

    m_logger->info("Local FFT brick is %d x %d x %d starting at [%d, %d, %d]",
		   m_own_size[0], m_own_size[1], m_own_size[2],
		   m_own_offset[0], m_own_offset[1], m_own_offset[2]);
}

void poisson_solver_psm::setup_remap()
{
    if(m_remap != NULL) {
	delete m_remap;
    }
    m_remap = new remap(m_iris,
			m_mesh->m_own_size, m_mesh->m_own_offset,
			m_own_size, m_own_offset, 1, 0);
}

void poisson_solver_psm::commit()
{
    if(m_dirty) {
	calculate_eigenvalues();
	setup_fft_grid();
	setup_remap();

	memory::destroy_1d(m_rho);
	memory::create_1d(m_rho, m_mesh->m_own_size[0] * 
			  m_mesh->m_own_size[1] *
			  m_mesh->m_own_size[2]);

	// 2 *, because it needs to contain complex numbers at some point
	memory::destroy_1d(m_scratch);
	memory::create_1d(m_scratch, 2*m_mesh->m_own_size[0]*m_mesh->m_own_size[1]*m_mesh->m_own_size[2]);

	m_dirty = false;
    }
}

void poisson_solver_psm::copy_rho_from_mesh()
{

    int nx = m_mesh->m_own_size[0];
    int ny = m_mesh->m_own_size[1];
    int nz = m_mesh->m_own_size[2];
    
    int i=0;
    for(int x = 0; x < nx; x++) {
	for(int y = 0; y < ny; y++) {
	    for(int z = 0; z < nz; z++) {
		m_rho[i++] = m_mesh->m_rho[x][y][z];
	    }
	}
    }
}

void poisson_solver_psm::solve()
{
    copy_rho_from_mesh();
    m_remap->perform(m_rho, m_rho, m_scratch);
}
