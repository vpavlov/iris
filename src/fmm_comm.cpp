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

//
// Using the multipole acceptance criteria it finds out which leafs are too close and so need to participate
// in P2P and sends the leaf metainfo and particle details to that neighbour.
//
void fmm::exchange_p2p_halo()
{
    timer tm;
    tm.start();
    
    int alien_index = 0;
    int alien_flag = IRIS_FMM_CELL_ALIEN_L1;
    for(int i=0;i<3;i++) {
    	MPI_Request cnt_req[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    	MPI_Request data_req[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };

    	for(int j=0;j<2;j++) {
    	    int rank = m_proc_grid->m_hood[i][j];
    	    if(rank < 0 ||                                     // no pbc and no neighbour in this dir
    	       rank == m_local_comm->m_rank ||                 // not same rank
    	       (j == 1 && rank == m_proc_grid->m_hood[i][0]))  // this rank was processed on the previous iteration (e.g. PBC, 2 in dir, 0 has 1 as both left and right neighbours)
    	    {
    		continue;
    	    }
    	    send_particles_to_neighbour(rank, m_border_parts + j, cnt_req+j, data_req+j);
    	}

    	for(int j=0;j<2;j++) {
    	    int rank = m_proc_grid->m_hood[i][1-j];
    	    if(rank < 0 ||                                     // no pbc and no neighbour in this dir
    	       rank == m_local_comm->m_rank ||                 // not same rank
    	       (j == 1 && rank == m_proc_grid->m_hood[i][1]))  // this rank was processed on the previous iteration (e.g. PBC, 2 in dir, 0 has 1 as both left and right neighbours)
	    {
		alien_index++;
		alien_flag *= 2;
		continue;
	    }
	    recv_particles_from_neighbour(rank, alien_index, alien_flag);
	    alien_index++;
	    alien_flag *= 2;
    	}
    	for(int j=0;j<2;j++) {
    	    MPI_Wait(cnt_req+j, MPI_STATUS_IGNORE);
    	    MPI_Wait(data_req+j, MPI_STATUS_IGNORE);
    	}
    }

    tm.stop();
    m_logger->time("Halo exchange wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::send_particles_to_neighbour(int rank, std::vector<xparticle_t> *out_sendbuf, MPI_Request *out_cnt_req, MPI_Request *out_data_req)
{
    int bl_count = border_leafs(rank);
    
    int part_count = 0;
    for(int i=0;i<bl_count;i++) {
	part_count += m_xcells[m_border_leafs[i]].num_children;
    }

    m_logger->info("Will be sending %d particles (from %d border leafs) to neighbour %d", part_count, bl_count, rank);
    
    MPI_Isend(&part_count, 1, MPI_INT, rank, IRIS_TAG_FMM_P2P_HALO_CNT, m_local_comm->m_comm, out_cnt_req);

    out_sendbuf->clear();
    for(int i=0;i<bl_count;i++) {
	cell_t *leaf = &m_xcells[m_border_leafs[i]];
	for(int j=0;j<leaf->num_children;j++) {
	    xparticle_t *ptr;
	    if(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		if(leaf->flags & IRIS_FMM_CELL_ALIEN_L1) {
		    ptr = m_xparticles[0];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN_L2) {
		    ptr = m_xparticles[1];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN_L3) {
		    ptr = m_xparticles[2];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN_L4) {
		    ptr = m_xparticles[3];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN_L5) {
		    ptr = m_xparticles[4];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN_L6) {
		    ptr = m_xparticles[5];
		}
		xparticle_t p(ptr[leaf->first_child + j].xyzq[0],
			      ptr[leaf->first_child + j].xyzq[1],
			      ptr[leaf->first_child + j].xyzq[2],
			      ptr[leaf->first_child + j].xyzq[3],
			      ptr[leaf->first_child + j].cellID);
		out_sendbuf->push_back(p);
	    }else {
		xparticle_t p(m_particles[leaf->first_child + j].xyzq[0],
			      m_particles[leaf->first_child + j].xyzq[1],
			      m_particles[leaf->first_child + j].xyzq[2],
			      m_particles[leaf->first_child + j].xyzq[3],
			      m_particles[leaf->first_child + j].cellID);
		out_sendbuf->push_back(p);
	    }
	}
    }
    
    MPI_Isend(out_sendbuf->data(), part_count*sizeof(xparticle_t), MPI_BYTE, rank, IRIS_TAG_FMM_P2P_HALO, m_local_comm->m_comm, out_data_req);
}

int fmm::border_leafs(int rank)
{
    m_border_leafs.clear();
    int start = cell_meta_t::offset_for_level(max_level());
    int end = m_tree_size;
    for(int n = start; n<end; n++) {  // iterate through all the leafs
	if(m_xcells[n].num_children == 0) {  // only full cells
	    continue;
	}
	
	bool send = false;
	
	iris_real dn = m_xcells[n].ses.r;
	iris_real cx = m_xcells[n].ses.c.r[0];
	iris_real cy = m_xcells[n].ses.c.r[1];
	iris_real cz = m_xcells[n].ses.c.r[2];
	
	for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	    if(send) { break; }
	    for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
		if(send) { break; }
		for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		    iris_real x = cx + ix * m_domain->m_global_box.xsize;
		    iris_real y = cy + iy * m_domain->m_global_box.ysize;
		    iris_real z = cz + iz * m_domain->m_global_box.zsize;
		    iris_real rn = m_domain->m_local_boxes[rank].distance_to(x, y, z);
		    if (2 * dn/rn < m_mac) {
			continue;
		    }
		    // D(n)/r(n) >= θ - this means that this cell is too close to the border
		    // and is needed by the other processor to do P2P
		    send = true;
		    break;
		}
	    }
	}
	
	if(send) {
	    m_border_leafs.push_back(n);
	}
    }
    return m_border_leafs.size();
}

void fmm::recv_particles_from_neighbour(int rank, int alien_index, int alien_flag)
{
    int part_count;
    MPI_Recv(&part_count, 1, MPI_INT, rank, IRIS_TAG_FMM_P2P_HALO_CNT, m_local_comm->m_comm, MPI_STATUS_IGNORE);

    m_logger->trace("Will be receiving %d paricles from neighbour %d", part_count, rank);
    
    memory::destroy_1d(m_xparticles[alien_index]);
    memory::create_1d(m_xparticles[alien_index], part_count);
    
    MPI_Recv(m_xparticles[alien_index], part_count*sizeof(xparticle_t), MPI_BYTE, rank, IRIS_TAG_FMM_P2P_HALO, m_local_comm->m_comm, MPI_STATUS_IGNORE);

    distribute_particles(m_xparticles[alien_index], part_count, alien_flag, m_xcells);
}

int fmm::comm_LET_cpu(cell_t *in_cells, iris_real *in_M)
{    
    // we're sending cell-in-transits, which are:
    //   - cellID (int)
    //   - ses (3 coordinates of centre + 1 radius)
    //   - m_nterms complex numbers for the multipole expansions
    int unit_size = sizeof(int) + sizeof(sphere_t) + 2*m_nterms*sizeof(iris_real);

    m_sendbuf = (unsigned char *)memory::wmalloc_cap(m_sendbuf, m_tree_size, unit_size, &m_sendbuf_cap);
    
    int hwm = 0;  // high-water-mark in sendbuf
    for(int rank=0;rank<m_local_comm->m_size;rank++) {
	if(rank == m_local_comm->m_rank) {
	    m_sendcnt[rank] = 0;
	    m_senddisp[rank] = hwm;
	    continue;
	}
	
	int num_cits = 0; // number of cells-in-transit	for this rank
	get_LET(rank, 0, m_sendbuf + hwm, unit_size, &num_cits, in_cells, in_M);
	m_sendcnt[rank] = num_cits;
	m_senddisp[rank] = hwm;
	hwm += num_cits * unit_size;
	m_sendbuf = (unsigned char *)memory::wrealloc_cap(m_sendbuf, hwm/unit_size + m_tree_size, unit_size, &m_sendbuf_cap);
    }

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

    m_recvbuf = (unsigned char *)memory::wmalloc_cap(m_recvbuf, rsize, 1, &m_recvbuf_cap);
    
    MPI_Alltoallv(m_sendbuf, m_sendcnt, m_senddisp, MPI_BYTE,
    		  m_recvbuf, m_recvcnt, m_recvdisp, MPI_BYTE,
    		  m_local_comm->m_comm);

    return rsize/unit_size;
}


void fmm::get_LET(int rank, int cellID, unsigned char *sendbuf, int unit_size, int *out_cits, cell_t *in_cells, iris_real *in_M)
{
    int level = cell_meta_t::level_of(cellID);

    int this_offset = cell_meta_t::offset_for_level(level);
    int children_offset = cell_meta_t::offset_for_level(level+1);

    int mask = IRIS_FMM_CELL_HAS_CHILD1;
    for(int i=0;i<8;i++) {
	if(!(in_cells[cellID].flags & mask)) {
	    mask <<= 1;
	    continue;
	}
	mask <<= 1;
	int childID = children_offset + (cellID-this_offset)*8 + i;
	
	bool is_close = (level < m_local_root_level);  // all cells above local root level are to be drilled-down

	// FIXME: discuss the implications of using the geometrical center instead of the real one...
	// iris_real dn = m_cell_meta[childID].maxr;
	// iris_real cx = m_cell_meta[childID].geomc[0];
	// iris_real cy = m_cell_meta[childID].geomc[1];
	// iris_real cz = m_cell_meta[childID].geomc[2];

	iris_real dn = in_cells[childID].ses.r;
	iris_real cx = in_cells[childID].ses.c.r[0];
	iris_real cy = in_cells[childID].ses.c.r[1];
	iris_real cz = in_cells[childID].ses.c.r[2];
	
	for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	    if(is_close) { break; }
	    for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
		if(is_close) { break; }
		for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		    iris_real x = cx + ix * m_domain->m_global_box.xsize;
		    iris_real y = cy + iy * m_domain->m_global_box.ysize;
		    iris_real z = cz + iz * m_domain->m_global_box.zsize;
		    iris_real rn = m_iris->m_domain->m_local_boxes[rank].distance_to(x, y, z);

		    if (2 * dn/rn < m_mac) {
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

	if(is_close && level < max_level()-1) {
	    get_LET(rank, childID, sendbuf, unit_size, out_cits, in_cells, in_M);
	}else {
	    memcpy(sendbuf + (*out_cits)*unit_size, &childID, sizeof(int));
	    memcpy(sendbuf + (*out_cits)*unit_size + sizeof(int), &(in_cells[childID].ses), sizeof(sphere_t));
	    memcpy(sendbuf + (*out_cits)*unit_size + sizeof(int) + sizeof(sphere_t), in_M + childID*2*m_nterms, 2*m_nterms*sizeof(iris_real));
	    *out_cits = *out_cits + 1;
	}
    }
}


void fmm::inhale_xcells(int in_count)
{
    int unit_size = sizeof(int) + sizeof(sphere_t) + 2*m_nterms*sizeof(iris_real);

    for(int i=0;i<in_count;i++) {
    	int cellID = *(int *)(m_recvbuf + unit_size * i);
	memcpy(&(m_xcells[cellID].ses), m_recvbuf + unit_size * i + sizeof(int), sizeof(sphere_t));
	memcpy(m_M + cellID*2*m_nterms, m_recvbuf + unit_size * i + sizeof(int) + sizeof(sphere_t), 2*m_nterms*sizeof(iris_real));
	assert(m_xcells[cellID].flags == 0);
	m_xcells[cellID].flags |= (IRIS_FMM_CELL_ALIEN_NL | IRIS_FMM_CELL_VALID_M);
    }
}
