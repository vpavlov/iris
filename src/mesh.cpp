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
#include <string.h>
#include <stdio.h>
#include "iris.h"
#include "mesh.h"
#include "proc_grid.h"
#include "memory.h"
#include "domain.h"
#include "logger.h"
#include "event.h"
#include "charge_assigner.h"
#include "comm_rec.h"
#include "tags.h"
#include "poisson_solver.h"

using namespace ORG_NCSA_IRIS;

mesh::mesh(iris *obj)
    :state_accessor(obj), m_size{0, 0, 0}, m_rho(NULL), m_rho_plus(NULL),
    m_dirty(true), m_initialized(false), m_phi(NULL),
    m_Ex(NULL), m_Ey(NULL), m_Ez(NULL)
{
}

mesh::~mesh()
{
    memory::destroy_3d(m_rho);
    memory::destroy_3d(m_rho_plus);
    memory::destroy_3d(m_phi);
    memory::destroy_3d(m_Ex);
    memory::destroy_3d(m_Ey);
    memory::destroy_3d(m_Ez);
}

void mesh::set_size(int nx, int ny, int nz)
{
    if(nx < 2 || ny < 2 || nz < 2) {
	throw std::invalid_argument("Invalid mesh size!");
    }

    m_size[0] = nx;
    m_size[1] = ny;
    m_size[2] = nz;

    m_initialized = true;
    m_dirty = true;
    m_logger->info("Discretization mesh is %d x %d x %d",
		    m_size[0], m_size[1], m_size[2]);
}

void mesh::commit()
{
    if(!m_domain->m_initialized) {
	throw std::logic_error("mesh commit called, but domain is not initialized!");
    }

    if(!m_initialized) {
	throw std::logic_error("mesh commit called without size being initialized!");
    }

    if(m_dirty) {
	m_h[0] = m_domain->m_global_box.xsize / m_size[0];
	m_h[1] = m_domain->m_global_box.ysize / m_size[1];
	m_h[2] = m_domain->m_global_box.zsize / m_size[2];

	m_hinv[0] = m_size[0] / m_domain->m_global_box.xsize;
	m_hinv[1] = m_size[1] / m_domain->m_global_box.ysize;
	m_hinv[2] = m_size[2] / m_domain->m_global_box.zsize;

	m_h3inv = m_hinv[0] * m_hinv[1] * m_hinv[2];

	m_own_size[0] = m_size[0] / m_proc_grid->m_size[0];
	m_own_size[1] = m_size[1] / m_proc_grid->m_size[1];
	m_own_size[2] = m_size[2] / m_proc_grid->m_size[2];
	
	int *c = m_proc_grid->m_coords;
	m_own_offset[0] = c[0] * m_own_size[0];
	m_own_offset[1] = c[1] * m_own_size[1];
	m_own_offset[2] = c[2] * m_own_size[2];
	
	memory::destroy_3d(m_rho);
	memory::create_3d(m_rho, m_own_size[0], m_own_size[1], m_own_size[2],
			  true);  // make sure ρ is cleared -- it's accumulating

	memory::destroy_3d(m_phi);
	memory::create_3d(m_phi, m_own_size[0], m_own_size[1], m_own_size[2]);

	memory::destroy_3d(m_Ex);
	memory::create_3d(m_Ex, m_own_size[0], m_own_size[1], m_own_size[2]);

	memory::destroy_3d(m_Ey);
	memory::create_3d(m_Ey, m_own_size[0], m_own_size[1], m_own_size[2]);

	memory::destroy_3d(m_Ez);
	memory::create_3d(m_Ez, m_own_size[0], m_own_size[1], m_own_size[2]);
	
	// ρ halo setup
	// extra cells in each direction: see what charge assigner requires
	// from below and from above
	int extra = m_chass->m_ics_to - m_chass->m_ics_from;

	// for odd orders, we have an additional layer at the right, since
	// charges closer to the right border should go for example
	// at (0, 1, 2) instead of (-1, 0, 1)
	if(m_chass->m_order % 2) {
	    extra++;
	}

	m_ext_size[0] = m_own_size[0] + extra;
	m_ext_size[1] = m_own_size[1] + extra;
	m_ext_size[2] = m_own_size[2] + extra;
	memory::destroy_3d(m_rho_plus);
	memory::create_3d(m_rho_plus, m_ext_size[0], m_ext_size[1],
			  m_ext_size[2],
			  true);  // make sure ρ is cleared -- it's accumulating

	
	// other configuration that depends on ours must be reset
	if(m_solver != NULL) {
	    m_solver->m_dirty = true;
	}

	m_dirty = false;

	m_logger->info("Local mesh is %d x %d x %d starting at [%d, %d, %d]",
			m_own_size[0], m_own_size[1], m_own_size[2],
			m_own_offset[0], m_own_offset[1], m_own_offset[2]);
	m_logger->info("Hx = %g, Hy = %g, Hz = %g", m_h[0], m_h[1], m_h[2]);
    }
}

void mesh::dump_bov(const char *fname, iris_real ***data)
{
    char values_fname[256];
    char header_fname[256];
    
    sprintf(values_fname, "%s-%d.bdata", fname, m_local_comm->m_rank);
    sprintf(header_fname, "%s-%d.bov", fname, m_local_comm->m_rank);
    
    // 1. write the bov file
    FILE *fp = fopen(values_fname, "wb");
    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
		fwrite(&(data[k][j][i]), sizeof(iris_real), 1, fp);
	    }
	}
    }
    fclose(fp);
    
    // 2. write the bov header
    fp = fopen(header_fname, "w");
    fprintf(fp, "TIME: 1.23456\n");
    fprintf(fp, "DATA_FILE: %s\n", values_fname);
    fprintf(fp, "DATA_SIZE: %d %d %d\n", m_own_size[0], m_own_size[1], m_own_size[2]);
    if(sizeof(iris_real) == sizeof(double)) {
	fprintf(fp, "DATA_FORMAT: DOUBLE\n");
    }else {
	fprintf(fp, "DATA_FORMAT: FLOAT\n");
    }
    fprintf(fp, "VARIABLE: DATA\n");
    fprintf(fp, "DATA_ENDIAN: LITTLE\n");
    fprintf(fp, "CENTERING: zonal\n");
    fprintf(fp, "BRICK_ORIGIN: %f %f %f\n",
	    m_domain->m_local_box.xlo, m_domain->m_local_box.ylo, m_domain->m_local_box.zlo);
    fprintf(fp, "BRICK_SIZE: %f %f %f\n",
	    m_domain->m_local_box.xsize, m_domain->m_local_box.ysize, m_domain->m_local_box.zsize);
    fclose(fp);
}

void mesh::dump_exyz(const char *fname)
{
    char values_fname[256];
    char header_fname[256];
    
    sprintf(values_fname, "%s-%d.bdata", fname, m_local_comm->m_rank);
    sprintf(header_fname, "%s-%d.bov", fname, m_local_comm->m_rank);
    
    // 1. write the bov file
    FILE *fp = fopen(values_fname, "wb");
    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
		fwrite(&(m_Ex[k][j][i]), sizeof(iris_real), 1, fp);
		fwrite(&(m_Ey[k][j][i]), sizeof(iris_real), 1, fp);
		fwrite(&(m_Ez[k][j][i]), sizeof(iris_real), 1, fp);
	    }
	}
    }
    fclose(fp);
    
    // 2. write the bov header
    fp = fopen(header_fname, "w");
    fprintf(fp, "TIME: 1.23456\n");
    fprintf(fp, "DATA_FILE: %s\n", values_fname);
    fprintf(fp, "DATA_SIZE: %d %d %d\n", m_own_size[0], m_own_size[1], m_own_size[2]);
    if(sizeof(iris_real) == sizeof(double)) {
	fprintf(fp, "DATA_FORMAT: DOUBLE\n");
    }else {
	fprintf(fp, "DATA_FORMAT: FLOAT\n");
    }
    fprintf(fp, "VARIABLE: E\n");
    fprintf(fp, "DATA_ENDIAN: LITTLE\n");
    fprintf(fp, "CENTERING: zonal\n");
    fprintf(fp, "BRICK_ORIGIN: %f %f %f\n",
	    m_domain->m_local_box.xlo, m_domain->m_local_box.ylo, m_domain->m_local_box.zlo);
    fprintf(fp, "BRICK_SIZE: %f %f %f\n",
	    m_domain->m_local_box.xsize, m_domain->m_local_box.ysize, m_domain->m_local_box.zsize);
    fprintf(fp, "DATA COMPONENTS: 3\n");
    fclose(fp);
}


void mesh::dump_ascii(const char *fname, iris_real ***data)
{
    char values_fname[256];
    char header_fname[256];
    
    sprintf(values_fname, "%s-%d.data", fname, m_local_comm->m_rank);
    
    // 1. write the bov file
    FILE *fp = fopen(values_fname, "wb");
    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
		fprintf(fp, "%.15g ", data[k][j][i]);
	    }
	}
    }
    fclose(fp);
}

void mesh::dump_log(const char *name, iris_real ***data)
{
    for(int i=0;i<m_own_size[0];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[2];k++) {
		m_logger->trace("%s[%d][%d][%d] = %.15g", name, i, j, k,
				data[i][j][k]);
	    }
	    m_logger->trace("");
	}
	m_logger->trace("");
    }
}

void mesh::assign_charges(iris_real *in_charges, int in_ncharges)
{
    box_t<iris_real> *gbox = &(m_domain->m_global_box);

    iris_real tmp = 0.0;

    for(int n=0;n<in_ncharges;n++) {
	iris_real tx=(in_charges[n*4+0]-gbox->xlo)*m_hinv[0]-m_own_offset[0];
	iris_real ty=(in_charges[n*4+1]-gbox->ylo)*m_hinv[1]-m_own_offset[1];
	iris_real tz=(in_charges[n*4+2]-gbox->zlo)*m_hinv[2]-m_own_offset[2];
	
	// the index of the cell that is to the "left" of the atom
	int ix = (int) (tx + m_chass->m_ics_bump);
	int iy = (int) (ty + m_chass->m_ics_bump);
	int iz = (int) (tz + m_chass->m_ics_bump);

	// distance (increasing to the left!) from the center of the
	// interpolation grid
	iris_real dx = ix - tx + m_chass->m_ics_center;
	iris_real dy = iy - ty + m_chass->m_ics_center;
	iris_real dz = iz - tz + m_chass->m_ics_center;

	m_chass->compute_weights(dx, dy, dz);

	iris_real t0 = m_mesh->m_h3inv * in_charges[n*4 + 3];  // q/V
	for(int i = 0; i < m_chass->m_order; i++) {
	    iris_real t1 = t0 * m_chass->m_weights[0][i];
	    for(int j = 0; j < m_chass->m_order; j++) {
		iris_real t2 = t1 * m_chass->m_weights[1][j];
		for(int k = 0; k < m_chass->m_order; k++) {
		    iris_real t3 = t2 * m_chass->m_weights[2][k];
		    m_rho_plus[ix+i][iy+j][iz+k] += t3;
		}
	    }
	}
    }
}

//
// This is how a line (let's say in X direction) of m_rho_plus looks like:
//
// |lll|ooooo|rrr|
//  \ / \   / \ /
//   A    B    C
// A = # of layers to send left;   = -m_chass->m_ics_from
// B = # of layers of my own mesh; = m_own_size
// C = # of layers to send right;  = m_chass->m_ics_to
// The routines below take this into account
//
// TODO: optimization: handle the case when peer is self
void mesh::send_rho_halo(int in_dim, int in_dir, iris_real **out_sendbuf,
			 MPI_Request *out_req)
{
    int A = -m_chass->m_ics_from;
    int C = m_chass->m_ics_to;
    if(m_chass->m_order % 2) {
	C++;
    }

    int sx, nx, ex;
    int sy, ny, ey;
    int sz, nz, ez;

    if(in_dim == 0) {
	if(in_dir == 0) {
	    sx = m_own_size[0] + A;
	    nx = C;
	}else {
	    sx = 0;
	    nx = A;
	}

	sy = 0;
	ny = m_ext_size[1];

	sz = 0;
	nz = m_ext_size[2];
    }else if(in_dim == 1) { 
	sx = 0;
	nx = m_ext_size[0];

	if(in_dir == 0) {
	    sy = m_own_size[1] + A;
	    ny = C;
	}else {
	    sy = 0;
	    ny = A;
	}

	sz = 0;
	nz = m_ext_size[2];
    }else {  
	sx = 0;
	nx = m_ext_size[0];

	sy = 0;
	ny = m_ext_size[1];

	if(in_dir == 0) {
	    sz = m_own_size[2] + A;
	    nz = C;
	}else {
	    sz = 0;
	    nz = A;
	}
    }

    ex = sx + nx;
    ey = sy + ny;
    ez = sz + nz;

    size_t size = nx*ny*nz*sizeof(iris_real);
    *out_sendbuf = (iris_real *)memory::wmalloc(size); 
    int n = 0;
    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		(*out_sendbuf)[n++] = m_rho_plus[i][j][k];
	    }
	}
    }
    m_iris->send_event(m_local_comm->m_comm,
		       m_proc_grid->m_hood[in_dim][in_dir],
		       IRIS_TAG_RHO_HALO + in_dim*2 + in_dir, size,
		       *out_sendbuf, out_req, NULL);
}

void mesh::recv_rho_halo(int in_dim, int in_dir)
{
    event_t ev;

    m_local_comm->get_event(m_proc_grid->m_hood[in_dim][1-in_dir],
			    IRIS_TAG_RHO_HALO + in_dim*2 + in_dir, ev);

    int A = -m_chass->m_ics_from;
    int C = m_chass->m_ics_to;
    if(m_chass->m_order % 2) {
	C++;
    }

    int sx, nx, ex;
    int sy, ny, ey;
    int sz, nz, ez;

    if(in_dim == 0) {
	if(in_dir == 0) {   // comes from left
	    sx = A;
	    nx = C;
	}else {
	    sx = m_own_size[0];
	    nx = A;
	}

	sy = 0;
	ny = m_ext_size[1];

	sz = 0;
	nz = m_ext_size[2];
    }else if(in_dim == 1) { 
	sx = 0;
	nx = m_ext_size[0];

	if(in_dir == 0) {
	    sy = A;
	    ny = C;
	}else {
	    sy = m_own_size[1];
	    ny = A;
	}

	sz = 0;
	nz = m_ext_size[2];
    }else {  
	sx = 0;
	nx = m_ext_size[0];

	sy = 0;
	ny = m_ext_size[1];

	if(in_dir == 0) {
	    sz = A;
	    nz = C;
	}else {
	    sz = m_own_size[2];
	    nz = A;
	}
    }

    ex = sx + nx;
    ey = sy + ny;
    ez = sz + nz;
    
    int n = 0;
    iris_real *data = (iris_real *)ev.data;
    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		m_rho_plus[i][j][k] += data[n++];
	    }
	}
    }
    
    memory::wfree(ev.data);
}

// The halo is exchanged; extract rho from rho_plus
void mesh::extract_rho()
{
    int sx, nx, ex;
    int sy, ny, ey;
    int sz, nz, ez;

    sx = sy = sz = -m_chass->m_ics_from;
    ex = sx + m_own_size[0];
    ey = sy + m_own_size[1];
    ez = sz + m_own_size[2];

    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		m_rho[i-sx][j-sy][k-sz] = m_rho_plus[i][j][k];
	    }
	}
    }
}

void mesh::exchange_rho_halo()
{
    iris_real *sendbufs[6];
    MPI_Request req[6];

    MPI_Barrier(m_local_comm->m_comm);

    send_rho_halo(0, 0, &sendbufs[0], &req[0]);
    send_rho_halo(0, 1, &sendbufs[1], &req[1]);
    recv_rho_halo(0, 0);
    recv_rho_halo(0, 1);

    send_rho_halo(1, 0, &sendbufs[2], &req[2]);
    send_rho_halo(1, 1, &sendbufs[3], &req[3]);
    recv_rho_halo(1, 0);
    recv_rho_halo(1, 1);

    send_rho_halo(2, 0, &sendbufs[4], &req[4]);
    send_rho_halo(2, 1, &sendbufs[5], &req[5]);
    recv_rho_halo(2, 0);
    recv_rho_halo(2, 1);

    extract_rho();

    MPI_Waitall(6, req, MPI_STATUSES_IGNORE);
    for(int i=0;i<6;i++) {
	memory::wfree(sendbufs[i]);
    }
}

void mesh::ijk_to_xyz(int i, int j, int k,
		      iris_real &x, iris_real &y, iris_real &z)
{
    x = m_domain->m_local_box.xlo + i * m_h[0];
    y = m_domain->m_local_box.ylo + j * m_h[1];
    z = m_domain->m_local_box.zlo + k * m_h[2];
}

void mesh::check_exyz()
{
    iris_real sum_ex = 0.0;
    iris_real sum_ey = 0.0;
    iris_real sum_ez = 0.0;

    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
		sum_ex += m_Ex[k][j][i];
		sum_ey += m_Ey[k][j][i];
		sum_ez += m_Ez[k][j][i];
	    }
	}
    }
    m_logger->trace("Ex sum = %g", sum_ex);
    m_logger->trace("Ey sum = %g", sum_ey);
    m_logger->trace("Ez sum = %g", sum_ez);
}

void mesh::exchange_field_halo()
{
    // Ex, Ey and Ez are calculated. Now, in order to interpolate them back
    // to the original atoms, each proc will need controbutions from its
    // neighbours.
    //
    // When exchanging ρ halo, we could store outgoing contributions while
    // we calculate charge assignment, collect them and then after all is done,
    // distribute them to the neighbours. Now, its different, because we need
    // incoming contribution from neighbours before we start interpolating the
    // fields back to the charges.
}
