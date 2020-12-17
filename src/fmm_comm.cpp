// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2021, the National Center for Supercomputing Applications
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
#include "fmm.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "comm_rec.h"
#include "proc_grid.h"
#include "tags.h"
#include "domain.h"
#include "logger.h"

using namespace ORG_NCSA_IRIS;

void fmm::exchange_rest_of_LET()
{
    
    // we're sending cell-in-transits, which are just
    //   - cellID (int)
    //   - m_nterms complex numbers for the multipole expansions
    int unit_size = sizeof(int) + 2*m_nterms*sizeof(iris_real);
    
    // create the buffer for the first rank
    unsigned char *sendbuf;
    memory::create_1d(sendbuf, m_tree_size * unit_size);

    int hwm = 0;  // high-water-mark in sendbuf
    for(int rank=0;rank<m_local_comm->m_size;rank++) {
	if(rank == m_local_comm->m_rank) {
	    m_sendcnt[rank] = 0;
	    m_senddisp[rank] = hwm;
	    continue;
	}
	
	int num_cits = 0; // number of cells-in-transit	for this rank
	get_LET(rank, 0, sendbuf + hwm, unit_size, &num_cits);
	m_sendcnt[rank] = num_cits;
	m_senddisp[rank] = hwm;
	hwm += num_cits * unit_size;
	sendbuf = (unsigned char *)memory::wrealloc(sendbuf, hwm + m_tree_size * unit_size);
    }

    sendbuf = (unsigned char *)memory::wrealloc(sendbuf, hwm);  // final size of the sendbuf

    for(int i=0;i<m_local_comm->m_size;i++) {
	m_logger->trace("sendcnt to rank %d = %d", i, m_sendcnt[i]);
    }
    
    MPI_Alltoall(m_sendcnt, 1 , MPI_INT, m_recvcnt, 1, MPI_INT, m_local_comm->m_comm);

    for(int i=0;i<m_local_comm->m_size;i++) {
	m_logger->trace("recvcnt from rank %d = %d", i, m_recvcnt[i]);
    }
    
    for(int i=0;i<m_local_comm->m_size;i++) {
	m_sendcnt[i] *= unit_size;
	m_recvcnt[i] *= unit_size;
    }
    
    int rsize = 0;
    for(int i=0;i<m_local_comm->m_size;i++) {
	m_recvdisp[i] = rsize;
	rsize += m_recvcnt[i];
    }

    unsigned char *recvbuf;
    memory::create_1d(recvbuf, rsize);

    MPI_Alltoallv(sendbuf, m_sendcnt, m_senddisp, MPI_BYTE,
    		  recvbuf, m_recvcnt, m_recvdisp, MPI_BYTE,
    		  m_local_comm->m_comm);

    inhale_xcells(recvbuf, rsize / unit_size);
    
    memory::destroy_1d(recvbuf);
    memory::destroy_1d(sendbuf);
}


void fmm::get_LET(int rank, int cellID, unsigned char *sendbuf, int unit_size, int *out_cits)
{
    int level = cell_meta_t::level_of(cellID);

    if(level == max_level()) {
	return;
    }

    int this_offset = cell_meta_t::offset_for_level(level);
    int children_offset = cell_meta_t::offset_for_level(level+1);

    int mask = IRIS_FMM_CELL_HAS_CHILD1;
    for(int i=0;i<8;i++) {
	if(!(m_cells[cellID].flags & mask)) {
	    mask <<= 1;
	    continue;
	}
	mask <<= 1;
	int childID = children_offset + (cellID-this_offset)*8 + i;
	
	bool is_close = (level < m_local_root_level);  // all cells above local root level are to be drilled-down
	
	iris_real dn = m_cells[childID].ses.r;
	iris_real cx = m_cells[childID].ses.c.r[0];
	iris_real cy = m_cells[childID].ses.c.r[1];
	iris_real cz = m_cells[childID].ses.c.r[2];
	
	for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	    if(is_close) { break; }
	    for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
		if(is_close) { break; }
		for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		    iris_real x = cx + ix * m_domain->m_global_box.xsize;
		    iris_real y = cy + iy * m_domain->m_global_box.ysize;
		    iris_real z = cz + iz * m_domain->m_global_box.zsize;
		    iris_real rn = m_iris->m_domain->m_local_boxes[rank].distance_to(x, y, z);

		    if (m_mac_let_corr * dn/rn < m_mac) {
			continue;
		    }
		    // D(n)/r(n) >= θ - this means that this cell is too close to the border
		    // It's worthless to send it, but we need to divide it and see if
		    // we need to send its children...
		    is_close = true;
		    break;
		}
	    }
	}

	if(is_close) {
	    get_LET(rank, childID, sendbuf, unit_size, out_cits);
	}else { // orig: not close OR close but leaf; BUT second option we already sent with partices ?!?
	    memcpy(sendbuf + (*out_cits)*unit_size, &childID, sizeof(int));
	    memcpy(sendbuf + (*out_cits)*unit_size + sizeof(int), m_M[childID], 2*m_nterms*sizeof(iris_real));
	    *out_cits = *out_cits + 1;
	}
    }
}


void fmm::inhale_xcells(unsigned char *recvbuf, int in_count)
{
    int unit_size = sizeof(int) + 2*m_nterms*sizeof(iris_real);

    for(int i=0;i<in_count;i++) {
    	int cellID = *(int *)(recvbuf + unit_size * i);
	memcpy(m_M[cellID], recvbuf + unit_size * i + sizeof(int), 2*m_nterms*sizeof(iris_real));
	m_xcells[cellID].flags |= (IRIS_FMM_CELL_ALIEN_NL | IRIS_FMM_CELL_VALID_M);
    }
}
