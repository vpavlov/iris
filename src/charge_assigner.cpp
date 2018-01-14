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
#include <stdexcept>
#include <vector>
#include <tuple>
#include "iris.h"
#include "charge_assigner.h"
#include "mesh.h"
#include "proc_grid.h"
#include "comm_rec.h"

#include "memory.h"
#include "domain.h"
#include "logger.h"
#include "event.h"

using namespace ORG_NCSA_IRIS;

static iris_real coeff2[2][2] =
    {{ 1/2.0,  1.0 },
     { 1/2.0, -1.0 }};

static iris_real coeff3[3][3] =
    {{ 1/8.0,  1/2.0,  1/2.0 },
     { 6/8.0,      0, -2/2.0 },
     { 1/8.0, -1/2.0,  1/2.0 }};

static iris_real coeff4[4][4] = 
    {{  1/48.0,  1/8.0,  1/4.0,  1/6.0 },
     { 23/48.0,  5/8.0, -1/4.0, -3/6.0 },
     { 23/48.0, -5/8.0, -1/4.0,  3/6.0 },
     {  1/48.0, -1/8.0,  1/4.0, -1/6.0 }};

static iris_real coeff5[5][5] =
    {{   1/384.0,   1/48.0,   1/16.0,  1/12.0,  1/24.0 },
     {  76/384.0,  22/48.0,   4/16.0, -2/12.0, -4/24.0 },
     { 230/384.0,        0, -10/16.0,       0,  6/24.0 },
     {  76/384.0, -22/48.0,   4/16.0,  2/12.0, -4/24.0 },
     {   1/384.0,  -1/48.0,   1/16.0, -1/12.0,  1/24.0 }};

static iris_real coeff6[6][6] =
    {{    1/3840.0,    1/384.0,   1/96.0,   1/48.0,  1/48.0,   1/120.0 },
     {  237/3840.0,   75/384.0,  21/96.0,   3/48.0, -3/48.0,  -5/120.0 },
     { 1682/3840.0,  154/384.0, -22/96.0, -14/48.0,  2/48.0,  10/120.0 },
     { 1682/3840.0, -154/384.0, -22/96.0,  14/48.0,  2/48.0, -10/120.0 },
     {  237/3840.0,  -75/384.0,  21/96.0,  -3/48.0, -3/48.0,   5/120.0 },
     {    1/3840.0,   -1/384.0,   1/96.0,  -1/48.0,  1/48.0,  -1/120.0 }};

static iris_real coeff7[7][7] =
    {{     1/46080.0,     1/3840.0,    1/768.0,   1/288.0,   1/192.0,  1/240.0,   1/720.0 },
     {   722/46080.0,   236/3840.0,   74/768.0,  20/288.0,   2/192.0, -4/240.0,  -6/720.0 },
     { 10543/46080.0,  1445/3840.0,   79/768.0, -43/288.0, -17/192.0,  5/240.0,  15/720.0 },
     { 23548/46080.0,            0, -308/768.0,         0,  28/192.0,        0, -20/720.0 },
     { 10543/46080.0, -1445/3840.0,   79/768.0,  43/288.0, -17/192.0, -5/240.0,  15/720.0 },
     {   722/46080.0,  -236/3840.0,   74/768.0, -20/288.0,   2/192.0,  4/240.0,  -6/720.0 },
     {     1/46080.0,    -1/3840.0,    1/768.0,  -1/288.0,   1/192.0, -1/240.0,   1/720.0 }};

charge_assigner::charge_assigner(iris *obj)
    :state_accessor(obj), m_order(0), m_dirty(true), m_weights(NULL)
{
}

charge_assigner::~charge_assigner()
{
    memory::destroy_2d(m_weights);
}

void charge_assigner::set_order(int in_order)
{
    if(in_order < 2 || in_order > 7) {
	throw std::invalid_argument("Orders below 2 and above 7 are not supported!");
    }

    m_order = in_order;
    m_dirty = true;
    m_logger->trace("Charge assignment/interpolation order set to %d", m_order);
}

void charge_assigner::commit()
{
    // set default values
    if(m_order == 0) {
	set_order(2);
    }

    if(m_dirty) {
	m_ics_from = -(m_order-1)/2;
	m_ics_to = m_order/2;
	
	if(m_order % 2) {
	    m_ics_bump = 0.5;
	    m_ics_center = 0.0;
	}else {
	    m_ics_bump = 0.0;
	    m_ics_center = 0.5;
	}
	
	if(m_order == 2) {
	    m_coeff = (iris_real *) coeff2;
	}else if(m_order == 3) {
	    m_coeff = (iris_real *) coeff3;
	}else if(m_order == 4) {
	    m_coeff = (iris_real *) coeff4;
	}else if(m_order == 5) {
	    m_coeff = (iris_real *) coeff5;
	}else if(m_order == 6) {
	    m_coeff = (iris_real *) coeff6;
	}else if(m_order == 7) {
	    m_coeff = (iris_real *) coeff7;
	}

	if(m_weights != NULL) {
	    memory::destroy_2d(m_weights);
	}
	memory::create_2d(m_weights, 3, m_order);
	m_dirty = false;
    }
}


void charge_assigner::compute_weights(iris_real dx, iris_real dy, iris_real dz)
{
    iris_real r1, r2, r3;

    for(int i = 0; i < m_order; i++) {
	r1 = r2 = r3 = (iris_real)0.0;
	
	for(int j = m_order - 1; j >= 0; j--) {
	    r1 = m_coeff[i*m_order + j] + r1 * dx;
	    r2 = m_coeff[i*m_order + j] + r2 * dy;
	    r3 = m_coeff[i*m_order + j] + r3 * dz;
	}
	
	m_weights[0][i] = r1;
	m_weights[1][i] = r2;
	m_weights[2][i] = r3;
    }
}

void charge_assigner::assign_charges(iris_real *atoms, int natoms)
{
    box_t<iris_real> *gbox = &(m_domain->m_global_box);
    iris_real *hinv = m_mesh->m_hinv;
    int *loffset = m_mesh->m_own_offset;
    int *lsize = m_mesh->m_own_size;
    iris_real ***rho = m_mesh->m_rho;

    for(int i=0;i<natoms;i++) {
	iris_real tx = (atoms[i*4 + 0] - gbox->xlo) * hinv[0] - loffset[0];
	iris_real ty = (atoms[i*4 + 1] - gbox->ylo) * hinv[1] - loffset[1];
	iris_real tz = (atoms[i*4 + 2] - gbox->zlo) * hinv[2] - loffset[2];

	// the number of the cell that is to the "left" of the atom
	int nx = (int) (tx + m_ics_bump);
	int ny = (int) (ty + m_ics_bump);
	int nz = (int) (tz + m_ics_bump);

	// distance (increasing to the left!) from the center of the interpolation grid
	iris_real dx = nx - tx + m_ics_center;
	iris_real dy = ny - ty + m_ics_center;
	iris_real dz = nz - tz + m_ics_center;

	compute_weights(dx, dy, dz);

	iris_real t0 = m_mesh->m_h3inv * atoms[i*4 + 3];  // charge/volume
	for(int x = 0; x < m_order; x++) {
	    iris_real t1 = t0 * m_weights[0][x];
	    for(int y = 0; y < m_order; y++) {
		iris_real t2 = t1 * m_weights[1][y];
		for(int z = 0; z < m_order; z++) {
		    iris_real t3 = t2 * m_weights[2][z];

		    int m_x = nx + x + m_ics_from;
		    int m_y = ny + y + m_ics_from;
		    int m_z = nz + z + m_ics_from;
		    
		    // figure out which neighbour we need to send this
		    // info to (if any)
		    int nidx = 0;
		    int ne_x = m_x;
		    int ne_y = m_y;
		    int ne_z = m_z;

		    if(m_x < 0) {
			// e.g. -1 becomes 127
			nidx += 1;
			ne_x = lsize[0] + m_x;
		    }else if(m_x >= lsize[0]) {
			nidx += 2;
			ne_x = m_x - lsize[0];      // e.g. 128 becomes 0
		    }

		    if(m_y < 0) {
			nidx += 3;
			ne_y = lsize[1] + m_y;
		    }else if(m_y >= lsize[1]) {
			nidx += 6;
			ne_y = m_y - lsize[1];
		    }

		    if(m_z < 0) {
			nidx += 9;
			ne_z = lsize[2] + m_z;
		    }else if(m_z >= lsize[2]) {
			nidx += 18;
			ne_z = m_z - lsize[2];
		    }

		    if(m_proc_grid->m_hood[nidx] != m_local_comm->m_rank) {
			std::tuple<int, int, int> entry = std::make_tuple(ne_x, ne_y, ne_z);
			m_halo[m_proc_grid->m_hood[nidx]][entry] += t3;
		    }else {
			rho[ne_x][ne_y][ne_z] += t3;
		    }
		}
	    }
	}
    }
}

// void charge_assigner::exchange_halo()
// {
//     m_iris->suspend_event_loop = true;
//     //MPI_Win_fence(MPI_MODE_NOPRECEDE, rho_win);
//     m_logger->trace("Staring halo exchange...");
//     for(auto it = outer_rho.begin(); it != outer_rho.end(); it++) {
//     	for(std::map<std::tuple<int, int, int>, iris_real>::iterator it2 = it->second.begin();
//     	    it2 != it->second.end();
//     	    it2++)
//     	{
//     	    int x = std::get<0>(it2->first);
//     	    int y = std::get<1>(it2->first);
//     	    int z = std::get<2>(it2->first);
//     	    int disp = z + lsize[2]*(y + lsize[1]*x);  // 3D row major order -> 1D

//     	    iris_real q = it2->second;
//     	    //MPI_Accumulate(&q, 1, IRIS_REAL, it->first, disp, 1, IRIS_REAL, MPI_SUM, rho_win);
//     	}
//     }
//     //MPI_Win_fence((MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), rho_win);
//     m_logger->trace("Halo exchange done");
//     m_iris->suspend_event_loop = false;
// }
