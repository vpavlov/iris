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

using namespace ORG_NCSA_IRIS;

mesh::mesh(iris *obj)
    :state_accessor(obj), m_size{0, 0, 0}, m_rho(NULL), m_dirty(true),
    m_initialized(false), m_halo(NULL)
{
}

mesh::~mesh()
{
    memory::destroy_3d(m_rho);
}

void mesh::set_size(int nx, int ny, int nz)
{
    if(nx <= 0 || ny <= 0 || nz <= 0) {
	throw std::invalid_argument("Invalid mesh size!");
    }

    m_size[0] = nx;
    m_size[1] = ny;
    m_size[2] = nz;

    m_initialized = true;
    m_dirty = true;
    m_logger->trace("Discretization mesh is %d x %d x %d",
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
	memory::create_3d(m_rho, m_own_size[0], m_own_size[1], m_own_size[2]);
	for(int i=0;i<m_own_size[0];i++) {
	    for(int j=0;j<m_own_size[1];j++) {
		for(int k=0;k<m_own_size[2];k++) {
		    m_rho[i][j][k] = 0.0;
		}
	    }
	}
	
	if(m_halo != NULL) {
	    delete [] m_halo;
	}

	m_halo = new std::map<std::tuple<int, int, int>, iris_real>[m_iris->m_server_size];

	m_dirty = false;

	m_logger->trace("Local discretization mesh is %d x %d x %d starting at [%d, %d, %d]",
			m_own_size[0], m_own_size[1], m_own_size[2],
			m_own_offset[0], m_own_offset[1], m_own_offset[2]);
    }
}

void mesh::dump_rho(char *fname)
{
    char values_fname[256];
    char header_fname[256];
    
    sprintf(values_fname, "%s-%d.data", fname, m_local_comm->m_rank);
    sprintf(header_fname, "%s-%d.bov", fname, m_local_comm->m_rank);
    
    // 1. write the bov file
    FILE *fp = fopen(values_fname, "wb");
    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
		fwrite(&(m_rho[k][j][i]), sizeof(iris_real), 1, fp);
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
    fprintf(fp, "VARIABLE: RHO\n");
    fprintf(fp, "DATA_ENDIAN: LITTLE\n");
    fprintf(fp, "CENTERING: zonal\n");
    fprintf(fp, "BRICK_ORIGIN: %f %f %f\n",
	    m_domain->m_local_box.xlo, m_domain->m_local_box.ylo, m_domain->m_local_box.zlo);
    fprintf(fp, "BRICK_SIZE: %f %f %f\n",
	    m_domain->m_local_box.xsize, m_domain->m_local_box.ysize, m_domain->m_local_box.zsize);
    fclose(fp);
}


// TODO: openmp
void mesh::assign_charges(iris_real *in_charges, int in_ncharges)
{
    box_t<iris_real> *gbox = &(m_domain->m_global_box);
    
    for(int i=0;i<in_ncharges;i++) {
	iris_real tx = (in_charges[i*4 + 0] - gbox->xlo) * m_hinv[0] - m_own_offset[0];
	iris_real ty = (in_charges[i*4 + 1] - gbox->ylo) * m_hinv[1] - m_own_offset[1];
	iris_real tz = (in_charges[i*4 + 2] - gbox->zlo) * m_hinv[2] - m_own_offset[2];
	
	// the number of the cell that is to the "left" of the atom
	int nx = (int) (tx + m_chass->m_ics_bump);
	int ny = (int) (ty + m_chass->m_ics_bump);
	int nz = (int) (tz + m_chass->m_ics_bump);

	// distance (increasing to the left!) from the center of the interpolation grid
	iris_real dx = nx - tx + m_chass->m_ics_center;
	iris_real dy = ny - ty + m_chass->m_ics_center;
	iris_real dz = nz - tz + m_chass->m_ics_center;

	m_chass->compute_weights(dx, dy, dz);

	iris_real t0 = m_mesh->m_h3inv * in_charges[i*4 + 3];  // charge/volume
	for(int x = 0; x < m_chass->m_order; x++) {
	    iris_real t1 = t0 * m_chass->m_weights[0][x];
	    for(int y = 0; y < m_chass->m_order; y++) {
		iris_real t2 = t1 * m_chass->m_weights[1][y];
		for(int z = 0; z < m_chass->m_order; z++) {
		    iris_real t3 = t2 * m_chass->m_weights[2][z];

		    int m_x = nx + x + m_chass->m_ics_from;
		    int m_y = ny + y + m_chass->m_ics_from;
		    int m_z = nz + z + m_chass->m_ics_from;
		    
		    // figure out which neighbour we need to send this
		    // info to (if any)
		    int nidx = 0;
		    int ne_x = m_x;
		    int ne_y = m_y;
		    int ne_z = m_z;

		    if(m_x < 0) {
			// e.g. -1 becomes 127
			nidx += 1;
			ne_x = m_own_size[0] + m_x;
		    }else if(m_x >= m_own_size[0]) {
			nidx += 2;
			ne_x = m_x - m_own_size[0];      // e.g. 128 becomes 0
		    }

		    if(m_y < 0) {
			nidx += 3;
			ne_y = m_own_size[1] + m_y;
		    }else if(m_y >= m_own_size[1]) {
			nidx += 6;
			ne_y = m_y - m_own_size[1];
		    }

		    if(m_z < 0) {
			nidx += 9;
			ne_z = m_own_size[2] + m_z;
		    }else if(m_z >= m_own_size[2]) {
			nidx += 18;
			ne_z = m_z - m_own_size[2];
		    }

		    if(m_proc_grid->m_hood[nidx] != m_local_comm->m_rank) {
			std::tuple<int, int, int> entry =
			    std::make_tuple(ne_x, ne_y, ne_z);
			m_halo[m_proc_grid->m_hood[nidx]][entry] += t3;
		    }else {
			m_rho[ne_x][ne_y][ne_z] += t3;
		    }
		}
	    }
	}
    }
}

void mesh::exchange_halo()
{
    MPI_Request *req = new MPI_Request[m_iris->m_server_size];
    iris_halo_item_t **sendbufs = new iris_halo_item_t *[m_iris->m_server_size];

    MPI_Win win;
    int *pending = m_iris->stos_fence_pending(&win);

    for(int peer=0;peer<m_iris->m_server_size;peer++) {
	req[peer] = MPI_REQUEST_NULL;
	sendbufs[peer] = NULL;

	std::map<std::tuple<int, int, int>, iris_real> map = m_halo[peer];
	int count = map.size();  // number of halo items
	int size = count * sizeof(iris_halo_item_t);  // in bytes
	if(count == 0) {
	    continue;
	}

	m_logger->trace("There are %d halo items for %d", count, peer);

	int i = 0;
	sendbufs[peer] = (iris_halo_item_t *)memory::wmalloc(size);
    	for(auto j = map.begin(); j != map.end(); j++) {
    	    sendbufs[peer][i].v = j->second;
    	    sendbufs[peer][i].x = std::get<0>(j->first);
    	    sendbufs[peer][i].y = std::get<1>(j->first);
    	    sendbufs[peer][i++].z = std::get<2>(j->first);
    	}

	m_iris->send_event(m_local_comm->m_comm, peer, IRIS_TAG_RHO_HALO,
			   size, sendbufs[peer], &req[peer], win);
    }

    m_iris->stos_process_pending(pending, win);

    MPI_Waitall(m_iris->m_server_size, req, MPI_STATUSES_IGNORE);
    for(int i=0;i<m_iris->m_server_size;i++) {
	memory::wfree(sendbufs[i]);
    }
}

//TODO: openmp
void mesh::add_halo_items(iris_halo_item_t *in_items, int in_nitems)
{
    for(int i=0;i<in_nitems;i++) {
    	m_rho[in_items[i].x][in_items[i].y][in_items[i].z] += in_items[i].v;
    }
}
