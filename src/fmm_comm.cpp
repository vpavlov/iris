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


/////////////////////////
// Exchange halo (CPU) //
/////////////////////////


void h_border_leafs(int nleafs, int nthreads, int offset, cell_t *m_xcells, iris_real m_let_corr, proc_grid *m_proc_grid, domain *m_domain, int rank, iris_real m_mac,
		    std::vector<int> &m_a2a_cell_cnt)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
	int from, to;
	setup_work_sharing(nleafs, nthreads, &from, &to);
	for(int i=from;i<to;i++) {
	    int n = i + offset;
	    if(m_xcells[n].num_children == 0) {
		continue;
	    }

	    bool send = false;
	    
	    iris_real dn = m_xcells[n].ses.r + m_let_corr;
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
		m_a2a_cell_cnt[i] = m_xcells[n].num_children;
	    }
	}
    }
}

void h_excl_scan(const int *in, int *out, int size, int initial)
{
    if (size > 0) {
	//#pragma omp simd reduction(inscan, +:initial)
	for (int i=0;i<size-1;i++) {
	    out[i] = initial;
	    //#pragma omp scan exclusive(initial)
	    initial += in[i];
	}
	out[size - 1] = initial;
    }
}

void h_fill_sendbuf(int nleafs, int nthreads, int offset, std::vector<int> &m_a2a_cell_cnt, std::vector<int> &m_a2a_cell_disp, xparticle_t *out_sendbuf,
		    particle_t *m_particles, cell_t *m_xcells)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(nleafs, nthreads, &from, &to);
	for(int i=from;i<to;i++) {
	    int n = i + offset;
	    int cnt = m_a2a_cell_cnt[i];
	    int disp = m_a2a_cell_disp[i];
	    particle_t *src = m_particles + m_xcells[n].first_child;
	    xparticle_t *dest = out_sendbuf + disp;
	    for(int j=0;j<cnt;j++) {
	    	dest->xyzq[0] = src->xyzq[0];
	    	dest->xyzq[1] = src->xyzq[1];
	    	dest->xyzq[2] = src->xyzq[2];
	    	dest->xyzq[3] = src->xyzq[3];
	    	dest->cellID = src->cellID;
	    	src++;
	    	dest++;
	    }
	}
    }
}

int fmm::collect_halo_for(int rank, int hwm)
{
    int start = cell_meta_t::offset_for_level(max_level());
    int end = m_tree_size;
    int nleafs = end - start;

    m_a2a_cell_cnt.assign(nleafs, 0);
    m_a2a_cell_disp.resize(nleafs);
    
    h_border_leafs(nleafs, m_iris->m_nthreads, start, m_xcells, m_let_corr, m_proc_grid, m_domain, rank, m_mac, m_a2a_cell_cnt);
    h_excl_scan(m_a2a_cell_cnt.data(), m_a2a_cell_disp.data(), nleafs, 0);
    int halo_size = m_a2a_cell_cnt[nleafs-1] + m_a2a_cell_disp[nleafs-1];
    m_a2a_sendbuf.resize(m_a2a_sendbuf.size() + halo_size);
    h_fill_sendbuf(nleafs, m_iris->m_nthreads, start, m_a2a_cell_cnt, m_a2a_cell_disp, m_a2a_sendbuf.data() + hwm, m_particles, m_xcells);
    return halo_size;
}

void fmm::exchange_p2p_halo_cpu()
{
    timer tm;
    tm.start();
    
    m_a2a_send_cnt.assign(m_local_comm->m_size, 0);
    m_a2a_send_disp.assign(m_local_comm->m_size, 0);
    m_a2a_recv_cnt.resize(m_local_comm->m_size);
    m_a2a_recv_disp.resize(m_local_comm->m_size);
    m_a2a_sendbuf.clear();

    int hwm = 0;
    for(int rank=0;rank<m_local_comm->m_size;rank++) {
    	if(rank == m_local_comm->m_rank) {
    	    continue;
    	}
    	int cnt = collect_halo_for(rank, hwm);
    	m_a2a_send_cnt[rank] = cnt;
    	m_a2a_send_disp[rank] = hwm;
    	hwm += cnt;
    }

    // for(int i=0;i<m_local_comm->m_size;i++) {
    // 	m_logger->info("Will be sending %d particles to %d, starting from %d", m_a2a_send_cnt[i], i, m_a2a_send_disp[i]);
    // }
    
    MPI_Alltoall(m_a2a_send_cnt.data(), 1, MPI_INT, m_a2a_recv_cnt.data(), 1, MPI_INT, m_local_comm->m_comm);

    int rsize = 0;
    for(int i=0;i<m_local_comm->m_size;i++) {
    	m_a2a_recv_disp[i] = rsize;
    	rsize += m_a2a_recv_cnt[i];
    }

    m_xparticles[0] = (xparticle_t *)memory::wmalloc_cap(m_xparticles[0], rsize, sizeof(xparticle_t), m_xparticles_cap);
    
    // for(int i=0;i<m_local_comm->m_size;i++) {
    // 	m_logger->info("Will be receiving %d particles from %d, starting from %d", m_a2a_recv_cnt[i], i, m_a2a_recv_disp[i]);
    // }

    for(int i=0;i<m_local_comm->m_size;i++) {
    	m_a2a_send_cnt[i] *= sizeof(xparticle_t);
    	m_a2a_send_disp[i] *= sizeof(xparticle_t);
    	m_a2a_recv_cnt[i] *= sizeof(xparticle_t);
    	m_a2a_recv_disp[i] *= sizeof(xparticle_t);
    }

    MPI_Alltoallv(m_a2a_sendbuf.data(), m_a2a_send_cnt.data(), m_a2a_send_disp.data(), MPI_BYTE,
    		  m_xparticles[0], m_a2a_recv_cnt.data(), m_a2a_recv_disp.data(), MPI_BYTE,
    		  MPI_COMM_WORLD);
    
    distribute_particles(m_xparticles[0], rsize, IRIS_FMM_CELL_ALIEN_L1, m_xcells);
    
    tm.stop();
    m_logger->time("Halo exchange wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}


//////////////
// Comm LET //
//////////////


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

	iris_real dn = in_cells[childID].ses.r + m_let_corr;
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
	m_xcells[cellID].flags |= (IRIS_FMM_CELL_ALIEN_NL | IRIS_FMM_CELL_VALID_M);
    }
}
