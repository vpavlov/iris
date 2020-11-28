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
    // worst case scenario - allocate memory for sending all the particles
    // NOTE: this cannot be in commit since we don't know the number of local particles at that point...
    // NOTE GPU: would benefit from some kind of buffer manager to not allocate this every time (if all this happens on the GPU)
    memory::destroy_1d(m_border_parts[0]);
    memory::create_1d(m_border_parts[0], m_nparticles);
    
    memory::destroy_1d(m_border_parts[1]);
    memory::create_1d(m_border_parts[1], m_nparticles);

    int alien_flag = IRIS_FMM_CELL_ALIEN1;
    for(int i=0;i<3;i++) {
    	MPI_Request cnt_req[2];
    	MPI_Request data_req[2];
	int part_count[2];
    	for(int j=0;j<2;j++) {
    	    cnt_req[j] = MPI_REQUEST_NULL;
    	    data_req[j] = MPI_REQUEST_NULL;
    	}
    	for(int j=0;j<2;j++) {
    	    int rank = m_proc_grid->m_hood[i][j];
    	    if(rank < 0 || rank > m_local_comm->m_size ||      // no pbc and no neighbour in this dir
    	       rank == m_local_comm->m_rank ||                 // not same rank
    	       (j == 1 && rank == m_proc_grid->m_hood[i][0]))  // this rank was processed on the previous iteration (e.g. PBC, 2 in dir, 0 has 1 as both left and right neighbours)
    	    {
    		continue;
    	    }
    	    send_particles_to_neighbour(rank, 
					IRIS_TAG_FMM_P2P_HALO_CNT+i*j, IRIS_TAG_FMM_P2P_HALO+i*j,
					part_count + j, m_border_parts[j],
					cnt_req + j, data_req + j);
    	}
    	for(int j=0;j<2;j++) {
    	    int rank = m_proc_grid->m_hood[i][j];
    	    if(rank < 0 || rank > m_local_comm->m_size ||      // no pbc and no neighbour in this dir
    	       rank == m_local_comm->m_rank ||                 // not same rank
    	       (j == 1 && rank == m_proc_grid->m_hood[i][0]))  // this rank was processed on the previous iteration (e.g. PBC, 2 in dir, 0 has 1 as both left and right neighbours)
    	    {
    		continue;
    	    }
	    recv_particles_from_neighbour(rank, IRIS_TAG_FMM_P2P_HALO_CNT+i*j, IRIS_TAG_FMM_P2P_HALO+i*j, i*j, alien_flag);
	    alien_flag *= 2;
    	}
    	for(int j=0;j<2;j++) {
    	    MPI_Wait(cnt_req+j, MPI_STATUS_IGNORE);
    	    MPI_Wait(data_req+j, MPI_STATUS_IGNORE);
    	}
    }
}

void fmm::send_particles_to_neighbour(int rank, int count_tag, int data_tag,
				      int *out_part_count, xparticle_t *&out_sendbuf,
				      MPI_Request *out_cnt_req, MPI_Request *out_data_req)
{
    int bl_count = border_leafs(rank);
    
    *out_part_count = 0;
    for(int i=0;i<bl_count;i++) {
	*out_part_count += m_xcells[m_border_leafs[i]].num_children;
    }

    MPI_Isend(out_part_count, 1, MPI_INT, rank, count_tag, m_local_comm->m_comm, out_cnt_req);

    int n = 0;
    for(int i=0;i<bl_count;i++) {
	cell_t *leaf = &m_xcells[m_border_leafs[i]];
	for(int j=0;j<leaf->num_children;j++) {
	    xparticle_t *ptr;
	    if(leaf->flags & IRIS_FMM_CELL_ALIEN1 ||
	       leaf->flags & IRIS_FMM_CELL_ALIEN2 ||
	       leaf->flags & IRIS_FMM_CELL_ALIEN3 ||
	       leaf->flags & IRIS_FMM_CELL_ALIEN4 ||
	       leaf->flags & IRIS_FMM_CELL_ALIEN5 ||
	       leaf->flags & IRIS_FMM_CELL_ALIEN6)
	    {
		if(leaf->flags & IRIS_FMM_CELL_ALIEN1) {
		    ptr = m_xparticles[0];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN2) {
		    ptr = m_xparticles[1];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN3) {
		    ptr = m_xparticles[2];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN4) {
		    ptr = m_xparticles[3];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN5) {
		    ptr = m_xparticles[4];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN6) {
		    ptr = m_xparticles[5];
		}
		out_sendbuf[n].xyzq[0] = ptr[leaf->first_child + j].xyzq[0];
		out_sendbuf[n].xyzq[1] = ptr[leaf->first_child + j].xyzq[1];
		out_sendbuf[n].xyzq[2] = ptr[leaf->first_child + j].xyzq[2];
		out_sendbuf[n].xyzq[3] = ptr[leaf->first_child + j].xyzq[3];
		out_sendbuf[n++].cellID = ptr[leaf->first_child + j].cellID;
	    }else {
		out_sendbuf[n].xyzq[0] = m_particles[leaf->first_child + j].xyzq[0];
		out_sendbuf[n].xyzq[1] = m_particles[leaf->first_child + j].xyzq[1];
		out_sendbuf[n].xyzq[2] = m_particles[leaf->first_child + j].xyzq[2];
		out_sendbuf[n].xyzq[3] = m_particles[leaf->first_child + j].xyzq[3];
		out_sendbuf[n++].cellID = m_particles[leaf->first_child + j].cellID;
	    }
	}
    }
    
    MPI_Isend(out_sendbuf, (*out_part_count)*sizeof(xparticle_t), MPI_BYTE, rank, data_tag, m_local_comm->m_comm, out_data_req);
}

int fmm::border_leafs(int rank)
{
    int send_count = 0;
    int start = cell_meta_t::offset_for_level(max_level());
    int end = m_tree_size;
    for(int n = start; n<end; n++) {
	if(m_xcells[n].num_children == 0) {
	    continue;
	}
	bool send = false;
	iris_real dn = m_cell_meta[n].radius;
	iris_real cx = m_cell_meta[n].center[0];
	iris_real cy = m_cell_meta[n].center[1];
	iris_real cz = m_cell_meta[n].center[2];

	for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	    if(send) { break; }
	    for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
		if(send) { break; }
		for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		    iris_real x = cx + ix * m_domain->m_global_box.xsize;
		    iris_real y = cy + iy * m_domain->m_global_box.ysize;
		    iris_real z = cz + iz * m_domain->m_global_box.zsize;
		    iris_real rn = m_local_boxes[rank].distance_to(x, y, z);
		    if (dn/rn < m_mac) {
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
	    m_border_leafs[send_count++] = n;
	}
    }
    return send_count;
}

void fmm::recv_particles_from_neighbour(int rank, int count_tag, int data_tag, int alien_index, int alien_flag)
{
    int part_count;
    MPI_Recv(&part_count, 1, MPI_INT, rank, count_tag, m_local_comm->m_comm, MPI_STATUS_IGNORE);

    memory::destroy_1d(m_xparticles[alien_index]);
    memory::create_1d(m_xparticles[alien_index], part_count);
    
    MPI_Recv(m_xparticles[alien_index], part_count*sizeof(xparticle_t), MPI_BYTE, rank, data_tag, m_local_comm->m_comm, MPI_STATUS_IGNORE);

    distribute_particles(m_xparticles[alien_index], part_count, alien_flag, m_xcells);
}


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
	sendbuf = (unsigned char *)memory::wrealloc(sendbuf, hwm + (num_cits + m_tree_size) * unit_size);
	hwm += num_cits * unit_size;
    }

    sendbuf = (unsigned char *)memory::wrealloc(sendbuf, hwm);  // final size of the sendbuf

    for(int i=0;i<m_local_comm->m_size;i++) {
	m_logger->info("sendcnt to rank %d = %d", i, m_sendcnt[i]);
    }
    
    MPI_Alltoall(m_sendcnt, 1 , MPI_INT, m_recvcnt, 1, MPI_INT, m_local_comm->m_comm);

    for(int i=0;i<m_local_comm->m_size;i++) {
	m_logger->info("recvcnt from rank %d = %d", i, m_recvcnt[i]);
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
    
    for(int i=0;i<8;i++) {
	int childID = children_offset + (cellID-this_offset)*8 + i;
	if(m_cells[childID].num_children == 0) {
	    continue;
	}
	
	bool is_close = (level < m_local_root_level);  // all cells above local root level are to be drilled-down
	
	iris_real dn = m_cell_meta[childID].radius;
	iris_real cx = m_cell_meta[childID].center[0];
	iris_real cy = m_cell_meta[childID].center[1];
	iris_real cz = m_cell_meta[childID].center[2];
	
	for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	    if(is_close) { break; }
	    for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
		if(is_close) { break; }
		for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		    iris_real x = cx + ix * m_domain->m_global_box.xsize;
		    iris_real y = cy + iy * m_domain->m_global_box.ysize;
		    iris_real z = cz + iz * m_domain->m_global_box.zsize;
		    iris_real rn = m_local_boxes[rank].distance_to(x, y, z);
		    if (dn/rn < m_mac) {
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
	assert(m_xcells[cellID].flags == 0);
	m_xcells[cellID].flags = IRIS_FMM_CELL_ALIEN0;
    }
}
