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
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "iris.h"
#include "mesh_gpu.h"
#include "proc_grid.h"
#include "memory.h"
#include "domain.h"
#include "logger.h"
#include "event.h"
#include "charge_assigner_gpu.h"
#include "comm_rec.h"
#include "real.h"
#include "tags.h"
#include "poisson_solver.h"
#include "openmp.h"
#include "haloex_gpu.h"
#include "timer.h"
#include "cuda_parameters.h"

using namespace ORG_NCSA_IRIS;

void mesh_gpu::dump_bov(const char *fname, iris_real ***data)
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

void mesh_gpu::dump_exyz(const char *fname)
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


void mesh_gpu::dump_ascii(const char *fname, iris_real ***data)
{
    char values_fname[256];
    char header_fname[256];
    
    sprintf(values_fname, "%s-%d.data", fname, m_local_comm->m_rank);
    
    // 1. write the bov file
    FILE *fp = fopen(values_fname, "wb");
    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
	      //fprintf(fp, "%g\n", data[k][j][i]);
	      fprintf(fp, "%s[%d][%d][%d] %.15f 0x%x\n", fname, k, j, i, data[k][j][i], *(int*)&(data[k][j][i]));
	    }
	}
    }
    fclose(fp);
}

void mesh_gpu::dump_log(const char *name, iris_real ***data)
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

void mesh_gpu::assign_charges_gpu(iris_real* sendbuff_gpu)
{
	int nitems = m_ext_size[0]*m_ext_size[1]*m_ext_size[2];
	// it might be optimized using cudaMemset ....
	memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(&(m_rho_plus[0][0][0]), nitems, (iris_real)0.0);
	memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(&(m_Ex_plus[0][0][0]), nitems, (iris_real)0.0);
	memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(&(m_Ey_plus[0][0][0]), nitems, (iris_real)0.0);
	memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(&(m_Ez_plus[0][0][0]), nitems, (iris_real)0.0);
	memory_set_kernel<<<get_NBlocks(2,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(sendbuff_gpu, 2, (iris_real)0.0);
	cudaDeviceSynchronize();
	
    for(auto it = m_ncharges.begin(); it != m_ncharges.end(); it++) {
		int ncharges = it->second;
		iris_real *charges = m_charges[it->first];

		assign_charges1(ncharges, charges, sendbuff_gpu);
    }

    // m_logger->trace("assign_charge calling MPI_Allreduce");
    // MPI_Allreduce(sendbuf, recvbuf, 2, IRIS_REAL, MPI_SUM, m_iris->server_comm());
    // m_logger->trace("assign_charge called MPI_Allreduce");
    // m_qtot = recvbuf[0];
    // m_q2tot = recvbuf[1];
}

__global__
void assign_charges1_kernel(iris_real *in_charges, iris_real ***m_rho_plus,
	iris_real lbox_xlo, iris_real lbox_ylo, iris_real lbox_zlo,
	iris_real m_h_2, iris_real ics_bump, iris_real ics_center,
	int order, iris_real *m_coeff, iris_real m_h3inv, iris_real *sendbuff_gpu)
{
	size_t ndx = IRIS_CUDA_INDEX(x);
    int chunk_size = IRIS_CUDA_CHUNK(x,n);
    size_t from = ndx*chunk_size;
	size_t to = MIN((ndx+1)*chunk_size,n);
	
	for(int n=from;n<to;n++) {
		iris_real tz=(in_charges[n*5+2]-lbox_zlo)/m_h_2;
		int iz = (int) (tz + ics_bump);

		iris_real tx=(in_charges[n*5+0]-lbox_xlo)/m_h[0];
		iris_real ty=(in_charges[n*5+1]-lbox_ylo)/m_h[1];

		int ix = (int) (tx + ics_bump);
		int iy = (int) (ty + ics_bump);

		iris_real dx = ix - tx + ics_center;
		iris_real dy = iy - ty + ics_center;
		iris_real dz = iz - tz + ics_center;
		
		iris_real weights[3][IRIS_MAX_ORDER];
		
		compute_weights_dev(dx, dy, dz, m_coeff, weights, order);

		iris_real t0 = m_h3inv * in_charges[n*5 + 3];

		for(int i = 0; i < order; i++) {
			iris_real t1 = t0 * weights[0][i];
			for(int j = 0; j < order; j++) {
				iris_real t2 = t1 * weights[1][j];
				for(int k = 0; k < order; k++) {
					iris_real t3 = t2 * weights[2][k];
					atomicAdd(&(m_rho_plus[ix+i][iy+j][iz+k]), t3);
				}
			}
		}

		iris_real q = in_charges[n*5 + 3];
		atomicAdd(&(sendbuff_gpu[0]), q);
		atomicAdd(&(sendbuff_gpu[1]), q*q);
	}
}


void mesh_gpu::assign_charges1(int in_ncharges, iris_real *in_charges, iris_real *sendbuff_gpu)
{
    box_t<iris_real> *gbox = &(m_domain->m_global_box);
    box_t<iris_real> *lbox = &(m_domain->m_local_box);

	assign_charges1_kernel<<<get_NBlocks(n1,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>
	(in_charges, m_rho_plus, lbox->xlo, lbox->ylo, lbox->zlo, m_h[2], 
	m_chass->m_ics_bump, m_chass->m_ics_center,
	m_chass->m_order,m_chass->m_coeff, m_h3inv, sendbuff_gpu);

}

//
// This is how a line (let's say in X direction) of m_rho_plus looks like:
//
// |lll|ooooo|rrr|
//  \ / \   / \ /
//   A    B    C
// A = # of layers to send left;   = -m_chass->m_ics_from
// B = # of layers of my own mesh_gpu; = m_own_size
// C = # of layers to send right;  = m_chass->m_ics_to
// The routines below take this into account
//
// TODO: optimization: handle the case when peer is self
void mesh_gpu::send_rho_halo(int in_dim, int in_dir, iris_real **out_sendbuf,
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
	int B = m_own_size[0];
	if(in_dir == 0) {
	    sx = A + B;
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
	int B = m_own_size[1];
	sx = 0;
	nx = m_ext_size[0];

	if(in_dir == 0) {
	    sy = A + B;
	    ny = C;
	}else {
	    sy = 0;
	    ny = A;
	}

	sz = 0;
	nz = m_ext_size[2];
    }else {  
	int B = m_own_size[2];
	sx = 0;
	nx = m_ext_size[0];

	sy = 0;
	ny = m_ext_size[1];

	if(in_dir == 0) {
	    sz = A + B;
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

void mesh_gpu::recv_rho_halo(int in_dim, int in_dir)
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
	int B = m_own_size[0];

	if(in_dir == 0) {   // comes from left
	    sx = A;
	    nx = C;
	}else {
	    sx = B;
	    nx = A;
	}

	sy = 0;
	ny = m_ext_size[1];

	sz = 0;
	nz = m_ext_size[2];
    }else if(in_dim == 1) { 
	int B = m_own_size[1];
	sx = 0;
	nx = m_ext_size[0];

	if(in_dir == 0) {
	    sy = A;
	    ny = C;
	}else {
	    sy = B;
	    ny = A;
	}

	sz = 0;
	nz = m_ext_size[2];
    }else {  
	int B = m_own_size[2];
	sx = 0;
	nx = m_ext_size[0];

	sy = 0;
	ny = m_ext_size[1];

	if(in_dir == 0) {
	    sz = A;
	    nz = C;
	}else {
	    sz = B;
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
void mesh_gpu::extract_rho()
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

__global__
void mesh_gpu::ijk_to_xyz(int i, int j, int k,
		      iris_real &x, iris_real &y, iris_real &z)
{
    x = m_domain->m_local_box.xlo + i * m_h[0];
    y = m_domain->m_local_box.ylo + j * m_h[1];
    z = m_domain->m_local_box.zlo + k * m_h[2];
}

// Copy Ex, Ey and Ez to inner regions of Ex_plus, etc., thus preparing
// to exchange halo for E
void mesh_gpu::imtract_field()
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
		m_Ex_plus[i][j][k] = m_Ex[i-sx][j-sy][k-sz];
		m_Ey_plus[i][j][k] = m_Ey[i-sx][j-sy][k-sz];
		m_Ez_plus[i][j][k] = m_Ez[i-sx][j-sy][k-sz];
	    }
	}
    }
}

// Copy Ex, Ey and Ez to inner regions of Ex_plus, etc., thus preparing
// to exchange halo for E
void mesh_gpu::imtract_phi()
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
		m_phi_plus[i][j][k] = m_phi[i-sx][j-sy][k-sz];
	    }
	}
    }
}

//
// This is how a line (let's say in X direction) of m_Ex_plus looks like:
//
// |000|eeeee|000|
//  \ / \   / \ /
//   A    B    C
// A = # of layers to receive from left;   = -m_chass->m_ics_from
// B = # of layers of my own mesh_gpu; = m_own_size
// C = # of layers to receive from right;  = m_chass->m_ics_to
// The routines below take this into account
//
//
// |x|YYeeeX|yy|
//
// TODO: optimization: handle the case when peer is self
void mesh_gpu::send_field_halo(int in_dim, int in_dir, iris_real **out_sendbuf,
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
	int B = m_own_size[0];
	if(in_dir == 0) {
	    sx = B;
	    nx = A;
	}else {
	    sx = A;
	    nx = C;
	}

	sy = 0;
	ny = m_ext_size[1];

	sz = 0;
	nz = m_ext_size[2];
    }else if(in_dim == 1) { 
	int B = m_own_size[1];

	sx = 0;
	nx = m_ext_size[0];

	if(in_dir == 0) {
	    sy = B;
	    ny = A;
	}else {
	    sy = A;
	    ny = C;
	}

	sz = 0;
	nz = m_ext_size[2];
    }else {  
	int B = m_own_size[2];
	sx = 0;
	nx = m_ext_size[0];

	sy = 0;
	ny = m_ext_size[1];

	if(in_dir == 0) {
	    sz = B;
	    nz = A;
	}else {
	    sz = A;
	    nz = C;
	}
    }

    ex = sx + nx;
    ey = sy + ny;
    ez = sz + nz;

    size_t size = 3*nx*ny*nz*sizeof(iris_real);
    *out_sendbuf = (iris_real *)memory::wmalloc(size); 
    int n = 0;
    for(int i=sx;i<ex;i++) {
	for(int j=sy;j<ey;j++) {
	    for(int k=sz;k<ez;k++) {
		(*out_sendbuf)[n++] = m_Ex_plus[i][j][k];
		(*out_sendbuf)[n++] = m_Ey_plus[i][j][k];
		(*out_sendbuf)[n++] = m_Ez_plus[i][j][k];
	    }
	}
    }
    m_iris->send_event(m_local_comm->m_comm,
		       m_proc_grid->m_hood[in_dim][in_dir],
		       IRIS_TAG_EX_HALO + in_dim*2 + in_dir, size,
		       *out_sendbuf, out_req, NULL);
}

void mesh_gpu::recv_field_halo(int in_dim, int in_dir)
{
    event_t ev;
	#error "buffer manager not implemented"
    m_local_comm->get_event(m_proc_grid->m_hood[in_dim][1-in_dir],
			    IRIS_TAG_EX_HALO + in_dim*2 + in_dir, ev);

    int A = -m_chass->m_ics_from;
    int C = m_chass->m_ics_to;
    if(m_chass->m_order % 2) {
	C++;
    }

    int sx, nx, ex;
    int sy, ny, ey;
    int sz, nz, ez;

    if(in_dim == 0) {
	int B = m_own_size[0];

	if(in_dir == 0) {   // comes from left
	    sx = 0;
	    nx = A;
	}else {
	    sx = A + B;
	    nx = C;
	}

	sy = 0;
	ny = m_ext_size[1];

	sz = 0;
	nz = m_ext_size[2];
    }else if(in_dim == 1) { 
	int B = m_own_size[1];
	sx = 0;
	nx = m_ext_size[0];

	if(in_dir == 0) {
	    sy = 0;
	    ny = A;
	}else {
	    sy = A + B;
	    ny = C;
	}

	sz = 0;
	nz = m_ext_size[2];
    }else {  
	int B = m_own_size[2];
	sx = 0;
	nx = m_ext_size[0];

	sy = 0;
	ny = m_ext_size[1];

	if(in_dir == 0) {
	    sz = 0;
	    nz = A;
	}else {
	    sz = A + B;
	    nz = C;
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
		m_Ex_plus[i][j][k] += data[n++];
		m_Ey_plus[i][j][k] += data[n++];
		m_Ez_plus[i][j][k] += data[n++];
	    }
	}
    }
    
    memory::wfree(ev.data);
}


void mesh_gpu::assign_forces(bool ad)
{
    bool include_energy_virial = true;  // send the energy and virial to only one of the clients; to the others send 0
    MPI_Comm comm = m_iris->client_comm();

    for(auto it = m_ncharges.begin(); it != m_ncharges.end(); it++) {
	int peer = it->first;
	int ncharges = it->second;
	int size = 7*sizeof(iris_real) +             // 1 real for the E(k) energy + 6 reals for the virial
	    ncharges * 4 * sizeof(iris_real);        // 4 reals for each charge: id, Fx, Fy, Fz
	iris_real *forces = (iris_real *)memory::wmalloc(size);

	if(include_energy_virial) {
	    forces[0] = m_iris->m_Ek;
	    forces[1] = m_iris->m_virial[0];
	    forces[2] = m_iris->m_virial[1];
	    forces[3] = m_iris->m_virial[2];
	    forces[4] = m_iris->m_virial[3];
	    forces[5] = m_iris->m_virial[4];
	    forces[6] = m_iris->m_virial[5];
	}else {
	    forces[0] = 0.0;
	    forces[1] = 0.0;
	    forces[2] = 0.0;
	    forces[3] = 0.0;
	    forces[4] = 0.0;
	    forces[5] = 0.0;
	    forces[6] = 0.0;
	}
	
	m_forces[peer] = forces;
	if(ad) {
	    assign_forces1_ad(ncharges, m_charges[it->first], forces);
	}else {
	    assign_forces1(ncharges, m_charges[it->first], forces);
	}

	MPI_Request req;
	m_iris->send_event(comm, peer, IRIS_TAG_FORCES, size, forces, &req, NULL);
	MPI_Request_free(&req);
	include_energy_virial = false;
    }
}

void mesh_gpu::assign_forces1(int in_ncharges, iris_real *in_charges,
			  iris_real *out_forces)
{
    box_t<iris_real> *gbox = &(m_domain->m_global_box);
    box_t<iris_real> *lbox = &(m_domain->m_local_box);

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(in_ncharges, m_iris->m_nthreads, &from, &to);

	for(int n=from;n<to;n++) {
	  iris_real tx=(in_charges[n*5+0]-lbox->xlo)/m_h[0];
	  iris_real ty=(in_charges[n*5+1]-lbox->ylo)/m_h[1];
	  iris_real tz=(in_charges[n*5+2]-lbox->zlo)/m_h[2];

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

	    iris_real ekx = 0.0;
	    iris_real eky = 0.0;
	    iris_real ekz = 0.0;
	    
	    for(int i = 0; i < m_chass->m_order; i++) {
		iris_real t1 = m_chass->m_weights[tid][0][i];
		for(int j = 0; j < m_chass->m_order; j++) {
		    iris_real t2 = t1 * m_chass->m_weights[tid][1][j];
		    for(int k = 0; k < m_chass->m_order; k++) {
			iris_real t3 = t2 * m_chass->m_weights[tid][2][k];
			ekx -= t3 * m_Ex_plus[ix+i][iy+j][iz+k];
			eky -= t3 * m_Ey_plus[ix+i][iy+j][iz+k];
			ekz -= t3 * m_Ez_plus[ix+i][iy+j][iz+k];
		    }
		}
	    }
	    
	    iris_real factor = in_charges[n*5 + 3] * m_units->ecf;
	    out_forces[7 + n*4 + 0] = in_charges[n*5 + 4]+0.333333333;  // id
	    out_forces[7 + n*4 + 1] = factor * ekx;
	    out_forces[7 + n*4 + 2] = factor * eky;
	    out_forces[7 + n*4 + 3] = factor * ekz;
	    m_logger->trace("out_forces %f %f %f %f in_charges %f %f %f",out_forces[7 + n*4 + 0],out_forces[7 + n*4 + 1],out_forces[7 + n*4 + 2],out_forces[7 + n*4 + 3],in_charges[n*5+0],in_charges[n*5+1],in_charges[n*5+2]);
	}
    }
}

void mesh_gpu::assign_forces1_ad(int in_ncharges, iris_real *in_charges,
			     iris_real *out_forces)
{
    box_t<iris_real> *gbox = &(m_domain->m_global_box);

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(in_ncharges, m_iris->m_nthreads, &from, &to);

	for(int n=from;n<to;n++) {
	    iris_real tx=(in_charges[n*5+0]-gbox->xlo)*m_hinv[0]-m_own_offset[0];
	    iris_real ty=(in_charges[n*5+1]-gbox->ylo)*m_hinv[1]-m_own_offset[1];
	    iris_real tz=(in_charges[n*5+2]-gbox->zlo)*m_hinv[2]-m_own_offset[2];
	    
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
	    m_chass->compute_dweights(dx, dy, dz);

	    iris_real ekx = 0.0;
	    iris_real eky = 0.0;
	    iris_real ekz = 0.0;
	    
	    for(int i = 0; i < m_chass->m_order; i++) {
		for(int j = 0; j < m_chass->m_order; j++) {
		    for(int k = 0; k < m_chass->m_order; k++) {
			ekx +=
			    m_chass->m_dweights[tid][0][i] *
			    m_chass->m_weights[tid][1][j] *
			    m_chass->m_weights[tid][2][k] *
			    m_phi_plus[ix+i][iy+j][iz+k];
			eky +=
			    m_chass->m_weights[tid][0][i] *
			    m_chass->m_dweights[tid][1][j] *
			    m_chass->m_weights[tid][2][k] *
			    m_phi_plus[ix+i][iy+j][iz+k];
			ekz +=
			    m_chass->m_weights[tid][0][i] *
			    m_chass->m_weights[tid][1][j] *
			    m_chass->m_dweights[tid][2][k] *
			    m_phi_plus[ix+i][iy+j][iz+k];
		    }
		}
	    }
	    ekx *= m_hinv[0];
	    eky *= m_hinv[1];
	    ekz *= m_hinv[2];
	    
	    iris_real factor = in_charges[n*5 + 3] * m_units->ecf;
	    out_forces[7 + n*4 + 0] = in_charges[n*5 + 4];  // id
	    out_forces[7 + n*4 + 1] = factor * ekx;
	    out_forces[7 + n*4 + 2] = factor * eky;
	    out_forces[7 + n*4 + 3] = factor * ekz;
	}
    }
}

