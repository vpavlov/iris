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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include <string.h>
#include <stdexcept>
#include "iris_gpu.h"
#include "remap_gpu.h"
#include "logger_gpu.h"
#include "memory.h"
#include "comm_rec_gpu.h"
#include "tags.h"
#include "utils.h"
#include "remap_item_gpu.h"
#include "remap_item_complex_permute_gpu.h"
#include "remap_item_complex_permute2_gpu.h"
#include "mesh_gpu.h"

#warning "buffer allocation (buffer_manager) and gpu dev synchronization NOT READY"

using namespace ORG_NCSA_IRIS;
remap_gpu::remap_gpu(class iris_gpu *obj,
		     int *in_from_offset, int *in_from_size, 
		     int *in_to_offset, int *in_to_size,
		     int in_unit_size,
		     int in_permute,
		     const char *in_name,
		     bool in_use_collective)
    : state_accessor_gpu(obj), m_send_plans(NULL), m_recv_plans(NULL), m_nsend(0),
      m_nrecv(0), m_use_collective(in_use_collective), m_collective_comm(MPI_COMM_NULL)
{
    m_name = strdup(in_name);
    m_from.xlo = in_from_offset[0];
    m_from.ylo = in_from_offset[1];
    m_from.zlo = in_from_offset[2];

    m_from.xsize = in_from_size[0];
    m_from.ysize = in_from_size[1];
    m_from.zsize = in_from_size[2];

    m_from.xhi = in_from_offset[0] + in_from_size[0] - 1;
    m_from.yhi = in_from_offset[1] + in_from_size[1] - 1;
    m_from.zhi = in_from_offset[2] + in_from_size[2] - 1;

    m_to.xlo = in_to_offset[0];
    m_to.ylo = in_to_offset[1];
    m_to.zlo = in_to_offset[2];

    m_to.xsize = in_to_size[0];
    m_to.ysize = in_to_size[1];
    m_to.zsize = in_to_size[2];

    m_to.xhi = in_to_offset[0] + in_to_size[0] - 1;
    m_to.yhi = in_to_offset[1] + in_to_size[1] - 1;
    m_to.zhi = in_to_offset[2] + in_to_size[2] - 1;

    m_comm_list = (int *)memory::wmalloc(m_local_comm->m_size * sizeof(int));
    m_comm_len = 0;
    
    // All processors need this info from all other processors

    box_t<int> *to = (box_t<int> *)memory::wmalloc(m_local_comm->m_size * sizeof(box_t<int>));
    box_t<int> *from = (box_t<int> *)memory::wmalloc(m_local_comm->m_size * sizeof(box_t<int>));
    MPI_Allgather(&m_from, sizeof(box_t<int>), MPI_BYTE,
		  from, sizeof(box_t<int>), MPI_BYTE, m_local_comm->m_comm);

    MPI_Allgather(&m_to, sizeof(box_t<int>), MPI_BYTE,
		  to, sizeof(box_t<int>), MPI_BYTE, m_local_comm->m_comm);

    //-------------
    // Sending part
    //-------------

    // Find out how many we need to send by counting the overlapping boxes
    int nsend = 0;
    for(int i=0;i<m_local_comm->m_size;i++) {

	box_t<int> overlap = m_from && to[i];  // find intersection
	if(overlap.xsize > 0 && overlap.ysize > 0 && overlap.zsize > 0) {
	    nsend++;
	}
    }

    m_logger->trace("%s nsend = %d", m_name, nsend);
    // For each overlapping box, create a plan to send and store it in the
    // array of plans
    if(nsend) {
	m_send_plans = new remap_item_gpu[nsend];
	nsend = 0;
	int iproc = m_local_comm->m_rank;
	for(int i=0;i<m_local_comm->m_size;i++) {

	    iproc++;  // make sure self is last
	    if(iproc == m_local_comm->m_size) {
		iproc = 0;
	    }

	    box_t<int> overlap = m_from && to[iproc];
	    if(overlap.xsize > 0 && overlap.ysize > 0 && overlap.zsize > 0) {
		m_comm_list[m_comm_len++] = iproc;
		m_send_plans[nsend].m_peer = iproc;
		m_send_plans[nsend].m_offset = in_unit_size *
		    ROW_MAJOR_OFFSET(overlap.xlo - m_from.xlo,
				     overlap.ylo - m_from.ylo,
				     overlap.zlo - m_from.zlo,
				     m_from.ysize,
				     m_from.zsize);
		m_send_plans[nsend].m_nx = overlap.xsize;
		m_send_plans[nsend].m_ny = overlap.ysize;
		m_send_plans[nsend].m_nz = in_unit_size * overlap.zsize;
		m_send_plans[nsend].m_stride_line = in_unit_size * m_from.zsize;
		m_send_plans[nsend].m_stride_plane = in_unit_size * m_from.ysize * m_from.zsize;
		m_send_plans[nsend].m_size = in_unit_size * overlap.xsize * overlap.ysize * overlap.zsize;
		m_send_plans[nsend++].m_bufloc = 0;
	    }
	}

	// if we're sending to self, don't count it if not using collective
	m_nsend = nsend;
	if(!m_use_collective && (m_send_plans[nsend-1].m_peer == m_local_comm->m_rank)) {
	    m_nsend--;
	}
    }

    //---------------
    // Receiving part
    //---------------

    // Find out how many we need to send by counting the overlapping boxes
    int nrecv = 0;
    for(int i=0;i<m_local_comm->m_size;i++) {
	box_t<int> overlap = m_to && from[i];  // find intersection
	if(overlap.xsize > 0 && overlap.ysize > 0 && overlap.zsize > 0) {
	    nrecv++;
	}
    }

    // For each overlapping box, create a plan to recv and store it in the
    // array of plans
    if(nrecv) {

	if(in_permute == 0) {
	    m_recv_plans = new remap_item_gpu[nrecv];
	}else if(in_permute == 1 && in_unit_size == 2) {
	    m_recv_plans = new remap_item_complex_permute_gpu[nrecv];
	}else if(in_permute == 2 && in_unit_size == 2) {
	    m_recv_plans = new remap_item_complex_permute2_gpu[nrecv];
	}else {
	    throw std::invalid_argument("Unimplemented combination of in_permute/in_unit_size!");
	}

	nrecv = 0;
	int bufloc = 0;
	int iproc = m_local_comm->m_rank;
	for(int i=0;i<m_local_comm->m_size;i++) {

	    iproc++;  // make sure self is last
	    if(iproc == m_local_comm->m_size) {
		iproc = 0;
	    }

	    box_t<int> overlap = m_to && from[iproc];
	    if(overlap.xsize > 0 && overlap.ysize > 0 && overlap.zsize > 0) {

		bool found = false;
		for (int j=0;j<m_comm_len;j++) {
		    if (m_comm_list[j] == iproc) {
			found = true;
		    }
		}
		if (!found) {
		    m_comm_list[m_comm_len++] = iproc;
		}
		
		
		m_recv_plans[nrecv].m_peer = iproc;
		
		if(in_permute == 0) {
		    m_recv_plans[nrecv].m_offset = in_unit_size *
			ROW_MAJOR_OFFSET(overlap.xlo - m_to.xlo,
					 overlap.ylo - m_to.ylo,
					 overlap.zlo - m_to.zlo,
					 m_to.ysize,
					 m_to.zsize);
		    m_recv_plans[nrecv].m_nx = overlap.xsize;
		    m_recv_plans[nrecv].m_ny = overlap.ysize;
		    m_recv_plans[nrecv].m_nz = in_unit_size * overlap.zsize;
		    m_recv_plans[nrecv].m_stride_line = in_unit_size * m_to.zsize;
		    m_recv_plans[nrecv].m_stride_plane = 
			in_unit_size * m_to.ysize * m_to.zsize;
		}else if(in_permute == 1) {
		    m_recv_plans[nrecv].m_offset = in_unit_size *
			ROW_MAJOR_OFFSET(overlap.zlo - m_to.zlo,
					 overlap.xlo - m_to.xlo,
					 overlap.ylo - m_to.ylo,
					 m_to.xsize,
					 m_to.ysize);
		    m_recv_plans[nrecv].m_nx = overlap.xsize;
		    m_recv_plans[nrecv].m_ny = overlap.ysize;
		    m_recv_plans[nrecv].m_nz = overlap.zsize;  // unit in unpack
		    m_recv_plans[nrecv].m_stride_line = in_unit_size * m_to.ysize;
		    m_recv_plans[nrecv].m_stride_plane = 
			in_unit_size * m_to.xsize * m_to.ysize;
		}else {
		    m_recv_plans[nrecv].m_offset = in_unit_size *
			ROW_MAJOR_OFFSET(overlap.ylo - m_to.ylo,
					 overlap.zlo - m_to.zlo,
					 overlap.xlo - m_to.xlo,
					 m_to.zsize,
					 m_to.xsize);
		    m_recv_plans[nrecv].m_nx = overlap.xsize;
		    m_recv_plans[nrecv].m_ny = overlap.ysize;
		    m_recv_plans[nrecv].m_nz = overlap.zsize;
		    m_recv_plans[nrecv].m_stride_line = in_unit_size * m_to.xsize;
		    m_recv_plans[nrecv].m_stride_plane = 
			in_unit_size * m_to.zsize * m_to.xsize;
		}

		m_recv_plans[nrecv].m_size = in_unit_size *
		    overlap.xsize * overlap.ysize * overlap.zsize;
		m_recv_plans[nrecv].m_bufloc = bufloc;
		bufloc += m_recv_plans[nrecv++].m_size;
	    }
	}

	// if we're recving from self, don't count it
	m_nrecv = nrecv;
	if(!m_use_collective && (m_recv_plans[nrecv-1].m_peer == m_local_comm->m_rank)) {
	    m_nrecv--;
	}

	m_self = true;
	if(nrecv == m_nrecv) {
	    m_self = false;
	}

    }

    if(m_use_collective) {
	bool appending = true;
	while(appending) {
	    int new_len = m_comm_len;
	    appending = false;
	    for(int i=0;i<m_comm_len;i++) {
		for(int j=0;j<m_local_comm->m_size;j++) {
		    box_t<int> overlap1 = from[m_comm_list[i]] && to[j];
		    if(overlap1.xsize > 0 && overlap1.ysize > 0 && overlap1.zsize > 0) {
			bool found = false;
			for(int k=0;k<new_len;k++) {
			    if(m_comm_list[k] == j) {
				found = true;
				break;
			    }
			}
			if(!found) {
			    m_comm_list[new_len++] = j;
			    appending = true;
			}
		    }

		    box_t<int> overlap2 = from[m_comm_list[i]] && to[j];
		    if(overlap2.xsize > 0 && overlap2.ysize > 0 && overlap2.zsize > 0) {
			bool found = false;
			for(int k=0;k<new_len;k++) {
			    if(m_comm_list[k] == j) {
				found = true;
				break;
			    }
			}
			if(!found) {
			    m_comm_list[new_len++] = j;
			    appending = true;
			}
		    }
		}
	    }
	    m_comm_len = new_len;
	}

	if(m_comm_len > 0) {
	    qsort_int(m_comm_list, m_comm_len);
	    m_comm_list = (int *)memory::wrealloc(m_comm_list, m_comm_len*sizeof(int));
	    MPI_Group local_group, collective_group;
	    MPI_Comm_group(m_local_comm->m_comm, &local_group);
	    MPI_Group_incl(local_group, m_comm_len, m_comm_list, &collective_group);
	    MPI_Comm_create(m_local_comm->m_comm, collective_group, &m_collective_comm);
	    MPI_Group_free(&local_group);
	    MPI_Group_free(&collective_group);
	}else {
	    MPI_Comm_create(m_local_comm->m_comm, MPI_GROUP_EMPTY, &m_collective_comm);
	}
    }
    
    int size = 0;
    for(int i=0;i<m_nsend;i++) {
	size = MAX(size, m_send_plans[i].m_size);
    }

    if(size != 0) {
	m_sendbuf = (iris_real *)memory::wmalloc(size * sizeof(iris_real));
    }else {
	m_sendbuf = NULL;
    }

    memory::wfree(to);
    memory::wfree(from);
}

remap_gpu::~remap_gpu()
{
    if(m_send_plans != NULL) {
	delete [] m_send_plans;
    }

    if(m_recv_plans != NULL) {
	delete [] m_recv_plans;
    }

    memory::wfree(m_sendbuf);
    memory::wfree(m_comm_list);
    if(m_collective_comm != MPI_COMM_NULL) {
	MPI_Comm_free(&m_collective_comm);
    }
}

void remap_gpu::perform_p2p(iris_real ***in_src, iris_real *in_dest, iris_real *in_buf)
{
    MPI_Request *req = new MPI_Request[m_nrecv];

    m_iris->m_logger->trace("%s m_nrecv = %d", m_name, m_nrecv);
    for(int i = 0; i < m_nrecv; i++) {
	remap_item_gpu *xi = &m_recv_plans[i];
	MPI_Irecv(&in_buf[xi->m_bufloc], xi->m_size, IRIS_REAL, xi->m_peer,
		  IRIS_TAG_REMAP, m_local_comm->m_comm, &req[i]);
    }

    for(int i = 0; i < m_nsend; i++) {
	remap_item_gpu *xi = &m_send_plans[i];
	xi->pack(in_src,xi->m_offset, m_sendbuf,0);
	MPI_Send(m_sendbuf, xi->m_size, IRIS_REAL, xi->m_peer,
		 IRIS_TAG_REMAP, m_local_comm->m_comm);
    }

    if(m_self) {
	remap_item_gpu *si = &m_send_plans[m_nsend];
	remap_item_gpu *ri = &m_recv_plans[m_nrecv];
	si->pack(in_src,si->m_offset, in_buf,ri->m_bufloc);
	ri->unpack(in_buf,ri->m_bufloc, in_dest, ri->m_offset);
    }

    for(int i = 0; i < m_nrecv; i++) {
	int j;
	MPI_Waitany(m_nrecv, req, &j, MPI_STATUS_IGNORE);
	remap_item_gpu *ri = &m_recv_plans[j];
	ri->unpack(in_buf, ri->m_bufloc, in_dest, ri->m_offset);
    }

    delete req;
}

void remap_gpu::perform_collective(iris_real ***in_src, iris_real *in_dest, iris_real *in_buf)
{
    if(m_comm_len <= 0) {
	return;
    }

    int isend;
    int irecv;
    int send_buff_size = 0;
    int recv_buff_size = 0;

    for(int i=0;i<m_nsend;i++) {
	send_buff_size += m_send_plans[i].m_size;
    }
    for(int i=0;i<m_nrecv;i++) {
	recv_buff_size += m_recv_plans[i].m_size;
    }

    iris_real *send_buff_gpu = (iris_real *)memory_gpu::wmalloc(send_buff_size * sizeof(iris_real));
    iris_real *recv_buff_gpu = (iris_real *)memory_gpu::wmalloc(recv_buff_size * sizeof(iris_real));

    iris_real *send_buff, *recv_buff;
    int *send_counts_gpu, *recv_counts_gpu, *send_offsets_gpu, *recv_offsets_gpu;

    if (memory_gpu::m_env_psp_cuda!=0)
	{
	    send_counts_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	    recv_counts_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	    send_offsets_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	    recv_offsets_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	} else {
	send_buff = (iris_real *)memory::wmalloc(send_buff_size * sizeof(iris_real));
    	recv_buff = (iris_real *)memory::wmalloc(recv_buff_size * sizeof(iris_real));
    }

    int *send_counts = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *recv_counts = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *send_offsets = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *recv_offsets = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *recv_map = (int *)memory::wmalloc(m_comm_len * sizeof(int));

    int offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	send_counts[i] = 0;
	send_offsets[i] = 0;
	for(int j=0;j<m_nsend;j++) {
	    remap_item_gpu *plan = &m_send_plans[j];
	    if(plan->m_peer == m_comm_list[i]) {
		send_counts[i] = plan->m_size;
		send_offsets[i] = offset;
		plan->pack(in_src,plan->m_offset, send_buff_gpu,offset);
		offset += plan->m_size;
		break;
	    }
	}
    }

    offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	recv_counts[i] = 0;
	recv_offsets[i] = 0;
	recv_map[i] = -1;
	for(int j=0;j<m_nrecv;j++) {
	    remap_item_gpu *plan = &m_recv_plans[j];
	    if(plan->m_peer == m_comm_list[i]) {
		recv_counts[i] = plan->m_size;
		recv_offsets[i] = offset;
		offset += plan->m_size;
		recv_map[i] = j;
		break;
	    }
	}
    }

    if (memory_gpu::m_env_psp_cuda!=0) {
	memory_gpu::sync_gpu_buffer(send_counts_gpu, send_counts, m_comm_len * sizeof(int));
	memory_gpu::sync_gpu_buffer(send_offsets_gpu, send_offsets, m_comm_len * sizeof(int));
	MPI_Alltoallv(send_buff_gpu, send_counts_gpu, send_offsets_gpu, IRIS_REAL,
		      recv_buff_gpu, recv_counts_gpu, recv_offsets_gpu, IRIS_REAL,
		      m_collective_comm);
	memory_gpu::sync_cpu_buffer(recv_counts, recv_counts_gpu, m_comm_len * sizeof(int));
	memory_gpu::sync_cpu_buffer(recv_offsets, recv_offsets_gpu, m_comm_len * sizeof(int));
    } else {
#warning "all to all goes using cumemcpy all2allv cumemcpy"

	memory_gpu::sync_cpu_buffer(send_buff, send_buff_gpu, send_buff_size*sizeof(iris_real));

	MPI_Alltoallv(send_buff, send_counts, send_offsets, IRIS_REAL,
		      recv_buff, recv_counts, recv_offsets, IRIS_REAL,
		      m_collective_comm);
	
	memory_gpu::sync_gpu_buffer(recv_buff_gpu, recv_buff, recv_buff_size*sizeof(iris_real));
    }
    offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	if(recv_map[i] != -1) {
	    remap_item_gpu *plan = &m_recv_plans[recv_map[i]];
	    plan->unpack(recv_buff_gpu, offset, in_dest, plan->m_offset);
	    offset += plan->m_size;
	}
    }

    if (memory_gpu::m_env_psp_cuda!=0)
	{
	    memory_gpu::wfree(send_offsets_gpu);
	    memory_gpu::wfree(recv_offsets_gpu);
	    memory_gpu::wfree(send_counts_gpu);
	    memory_gpu::wfree(recv_counts_gpu);
	} else {
	memory::wfree(send_buff);
	memory::wfree(recv_buff);
    }
    memory::wfree(send_counts);
    memory::wfree(recv_counts);
    memory::wfree(send_offsets);
    memory::wfree(recv_offsets);
    memory::wfree(recv_map);
    memory_gpu::wfree(send_buff_gpu);
    memory_gpu::wfree(recv_buff_gpu);
}

/// source pointer 1d ///

void remap_gpu::perform_p2p(iris_real *in_src, iris_real *in_dest, iris_real *in_buf)
{
    MPI_Request *req = new MPI_Request[m_nrecv];

    m_iris->m_logger->trace("%s m_nrecv = %d", m_name, m_nrecv);
    for(int i = 0; i < m_nrecv; i++) {
	remap_item_gpu *xi = &m_recv_plans[i];
	MPI_Irecv(&in_buf[xi->m_bufloc], xi->m_size, IRIS_REAL, xi->m_peer,
		  IRIS_TAG_REMAP, m_local_comm->m_comm, &req[i]);
    }

    for(int i = 0; i < m_nsend; i++) {
	remap_item_gpu *xi = &m_send_plans[i];
	xi->pack(in_src,xi->m_offset, m_sendbuf,0);
	MPI_Send(m_sendbuf, xi->m_size, IRIS_REAL, xi->m_peer,
		 IRIS_TAG_REMAP, m_local_comm->m_comm);
    }

    if(m_self) {
	remap_item_gpu *si = &m_send_plans[m_nsend];
	remap_item_gpu *ri = &m_recv_plans[m_nrecv];
	si->pack(in_src,si->m_offset, in_buf,ri->m_bufloc);
	ri->unpack(in_buf,ri->m_bufloc, in_dest, ri->m_offset);
    }

    for(int i = 0; i < m_nrecv; i++) {
	int j;
	MPI_Waitany(m_nrecv, req, &j, MPI_STATUS_IGNORE);
	remap_item_gpu *ri = &m_recv_plans[j];
	ri->unpack(in_buf, ri->m_bufloc, in_dest, ri->m_offset);
    }

    delete req;
}

void remap_gpu::perform_collective(iris_real *in_src, iris_real *in_dest, iris_real *in_buf)
{
    if(m_comm_len <= 0) {
	return;
    }

    int isend;
    int irecv;
    int send_buff_size = 0;
    int recv_buff_size = 0;

    for(int i=0;i<m_nsend;i++) {
	send_buff_size += m_send_plans[i].m_size;
    }
    for(int i=0;i<m_nrecv;i++) {
	recv_buff_size += m_recv_plans[i].m_size;
    }

    iris_real *send_buff_gpu = (iris_real *)memory_gpu::wmalloc(send_buff_size * sizeof(iris_real));
    iris_real *recv_buff_gpu = (iris_real *)memory_gpu::wmalloc(recv_buff_size * sizeof(iris_real));

    iris_real *send_buff, *recv_buff;
    int *send_counts_gpu, *recv_counts_gpu, *send_offsets_gpu, *recv_offsets_gpu;

    if (memory_gpu::m_env_psp_cuda!=0) {
	send_counts_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	recv_counts_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	send_offsets_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	recv_offsets_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
    }else {
	send_buff = (iris_real *)memory::wmalloc(send_buff_size * sizeof(iris_real));
    	recv_buff = (iris_real *)memory::wmalloc(recv_buff_size * sizeof(iris_real));
    }

    int *send_counts = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *recv_counts = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *send_offsets = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *recv_offsets = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    int *recv_map = (int *)memory::wmalloc(m_comm_len * sizeof(int));

    int offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	send_counts[i] = 0;
	send_offsets[i] = 0;
	for(int j=0;j<m_nsend;j++) {
	    remap_item_gpu *plan = &m_send_plans[j];
	    if(plan->m_peer == m_comm_list[i]) {
		send_counts[i] = plan->m_size;
		send_offsets[i] = offset;
		plan->pack(in_src,plan->m_offset, send_buff_gpu,offset);
		offset += plan->m_size;
		break;
	    }
	}
    }

    offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	recv_counts[i] = 0;
	recv_offsets[i] = 0;
	recv_map[i] = -1;
	for(int j=0;j<m_nrecv;j++) {
	    remap_item_gpu *plan = &m_recv_plans[j];
	    if(plan->m_peer == m_comm_list[i]) {
		recv_counts[i] = plan->m_size;
		recv_offsets[i] = offset;
		offset += plan->m_size;
		recv_map[i] = j;
		break;
	    }
	}
    }

    sync_with_gpu();

    if (memory_gpu::m_env_psp_cuda!=0) {
	memory_gpu::sync_gpu_buffer(send_counts_gpu, send_counts, m_comm_len * sizeof(int));
	memory_gpu::sync_gpu_buffer(send_offsets_gpu, send_offsets, m_comm_len * sizeof(int));
	MPI_Alltoallv(send_buff_gpu, send_counts_gpu, send_offsets_gpu, IRIS_REAL,
		      recv_buff_gpu, recv_counts_gpu, recv_offsets_gpu, IRIS_REAL,
		      m_collective_comm);
	memory_gpu::sync_cpu_buffer(recv_counts, recv_counts_gpu, m_comm_len * sizeof(int));
	memory_gpu::sync_cpu_buffer(recv_offsets, recv_offsets_gpu, m_comm_len * sizeof(int));
    } else {
#warning "all to all goes using cumemcpy all2allv cumemcpy"

	memory_gpu::sync_cpu_buffer(send_buff, send_buff_gpu, send_buff_size*sizeof(iris_real));
		
	MPI_Alltoallv(send_buff, send_counts, send_offsets, IRIS_REAL,
		      recv_buff, recv_counts, recv_offsets, IRIS_REAL,
		      m_collective_comm);
	//	m_mesh->dump_ascii("after-a2a-sync",&(recv_buff[0]),recv_buff_size);
	memory_gpu::sync_gpu_buffer(recv_buff_gpu, recv_buff, recv_buff_size*sizeof(iris_real));
    }

    offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	if(recv_map[i] != -1) {
	    remap_item_gpu *plan = &m_recv_plans[recv_map[i]];
	    plan->unpack(recv_buff_gpu, offset, in_dest, plan->m_offset);
	    offset += plan->m_size;
	}
    }

    sync_with_gpu();

    if (memory_gpu::m_env_psp_cuda!=0) {
	memory_gpu::wfree(send_offsets_gpu);
	memory_gpu::wfree(recv_offsets_gpu);
	memory_gpu::wfree(send_counts_gpu);
	memory_gpu::wfree(recv_counts_gpu);
    } else {
	memory::wfree(send_buff);
	memory::wfree(recv_buff);
    }
    memory::wfree(send_counts);
    memory::wfree(recv_counts);
    memory::wfree(send_offsets);
    memory::wfree(recv_offsets);
    memory::wfree(recv_map);
    memory_gpu::wfree(send_buff_gpu);
    memory_gpu::wfree(recv_buff_gpu);
}

// ========================== async ===================================

// ==================== init collective remap =========================
void remap_gpu::perform_collective_init(iris_real *in_src, iris_real *in_dest, collective_fft3D_state &fftstate)
{
    if(m_comm_len <= 0) {
	return;
    }
	
    fftstate.send_buff_size = 0;
    fftstate.recv_buff_size = 0;

    for(int i=0;i<m_nsend;i++) {
	fftstate.send_buff_size += m_send_plans[i].m_size;
    }
    for(int i=0;i<m_nrecv;i++) {
	fftstate.recv_buff_size += m_recv_plans[i].m_size;
    }
	
    free_collective_fft3D_memory(fftstate);
    fftstate.send_buff_gpu = (iris_real *)memory_gpu::wmalloc(fftstate.send_buff_size * sizeof(iris_real));
    fftstate.recv_buff_gpu = (iris_real *)memory_gpu::wmalloc(fftstate.recv_buff_size * sizeof(iris_real));

    // iris_real *send_buff, *recv_buff;
    // int *send_counts_gpu, *recv_counts_gpu, *send_offsets_gpu, *recv_offsets_gpu;

    if (memory_gpu::m_env_psp_cuda!=0) {
	fftstate.send_counts_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	fftstate.recv_counts_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	fftstate.send_offsets_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
	fftstate.recv_offsets_gpu = (int *)memory_gpu::wmalloc(m_comm_len * sizeof(int));
    }else {
	//fftstate.send_buff = (iris_real *)memory::wmalloc(fftstate.send_buff_size * sizeof(iris_real));
	cudaHostAlloc(&(fftstate.send_buff),fftstate.send_buff_size * sizeof(iris_real),0);
    	//fftstate.recv_buff = (iris_real *)memory::wmalloc(fftstate.recv_buff_size * sizeof(iris_real));
	cudaHostAlloc(&(fftstate.recv_buff),fftstate.recv_buff_size * sizeof(iris_real),0);
    }

    fftstate.send_counts = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    fftstate.recv_counts = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    fftstate.send_offsets = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    fftstate.recv_offsets = (int *)memory::wmalloc(m_comm_len * sizeof(int));
    fftstate.recv_map = (int *)memory::wmalloc(m_comm_len * sizeof(int));
}

void remap_gpu::perform_collective_pack(iris_real *in_src, iris_real *in_dest, collective_fft3D_state &fftstate)
{
    int offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	fftstate.send_counts[i] = 0;
	fftstate.send_offsets[i] = 0;
	for(int j=0;j<m_nsend;j++) {
	    remap_item_gpu *plan = &m_send_plans[j];
	    if(plan->m_peer == m_comm_list[i]) {
		fftstate.send_counts[i] = plan->m_size;
		fftstate.send_offsets[i] = offset;
		plan->pack(in_src,plan->m_offset, fftstate.send_buff_gpu,offset,fftstate.gpu_stream);
		offset += plan->m_size;
		break;
	    }
	}
    }

    offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	fftstate.recv_counts[i] = 0;
	fftstate.recv_offsets[i] = 0;
	fftstate.recv_map[i] = -1;
	for(int j=0;j<m_nrecv;j++) {
	    remap_item_gpu *plan = &m_recv_plans[j];
	    if(plan->m_peer == m_comm_list[i]) {
		fftstate.recv_counts[i] = plan->m_size;
		fftstate.recv_offsets[i] = offset;
		offset += plan->m_size;
		fftstate.recv_map[i] = j;
		break;
	    }
	}
    }



}

void remap_gpu::perform_collective_communicate1(collective_fft3D_state &fftstate)
{
    if(m_comm_len <= 0) {
	return;
    }
    // exchange buffers
    if (memory_gpu::m_env_psp_cuda!=0) {

	memory_gpu::sync_gpu_buffer(fftstate.send_counts_gpu, fftstate.send_counts, m_comm_len * sizeof(int),fftstate.gpu_stream);
	memory_gpu::sync_gpu_buffer(fftstate.send_offsets_gpu, fftstate.send_offsets, m_comm_len * sizeof(int),fftstate.gpu_stream);
	
    } else {

	memory_gpu::sync_cpu_buffer(fftstate.send_buff, fftstate.send_buff_gpu, fftstate.send_buff_size*sizeof(iris_real),fftstate.gpu_stream);
		
    }
}

// ==================== do collective communication =========================

void remap_gpu::perform_collective_communicate(collective_fft3D_state &fftstate)
{
    if(m_comm_len <= 0) {
	return;
    }

    cudaStreamSynchronize(fftstate.gpu_stream);
    printf("communicate 0x%x\n",fftstate.gpu_stream);
    // do all2all
    if (memory_gpu::m_env_psp_cuda!=0) {

	MPI_Alltoallv(fftstate.send_buff_gpu, fftstate.send_counts_gpu, fftstate.send_offsets_gpu, IRIS_REAL,
		      fftstate.recv_buff_gpu, fftstate.recv_counts_gpu, fftstate.recv_offsets_gpu, IRIS_REAL,
		      m_collective_comm);

    } else {

	MPI_Alltoallv(fftstate.send_buff, fftstate.send_counts, fftstate.send_offsets, IRIS_REAL,
		      fftstate.recv_buff, fftstate.recv_counts, fftstate.recv_offsets, IRIS_REAL,
		      m_collective_comm);

    }

}

void remap_gpu::perform_collective_finalize1(iris_real *in_dest, collective_fft3D_state &fftstate)
{
    if(m_comm_len <= 0) {
	return;
    }
    //cudaStreamSynchronize(fftstate.gpu_stream);
    printf("sync buffs 0x%x\n",fftstate.gpu_stream);
    // exchange buffers
    if (memory_gpu::m_env_psp_cuda!=0) {

	memory_gpu::sync_cpu_buffer(fftstate.recv_counts, fftstate.recv_counts_gpu, m_comm_len * sizeof(int),fftstate.gpu_stream);
	memory_gpu::sync_cpu_buffer(fftstate.recv_offsets, fftstate.recv_offsets_gpu, m_comm_len * sizeof(int),fftstate.gpu_stream);

    } else {
	//m_mesh->dump_ascii("after-a2a-async",&(fftstate.recv_buff[0]),fftstate.recv_buff_size);
	memory_gpu::sync_gpu_buffer(fftstate.recv_buff_gpu, fftstate.recv_buff, fftstate.recv_buff_size*sizeof(iris_real),fftstate.gpu_stream);

    }
}

// ==================== unpack and free the buffers =========================

void remap_gpu::perform_collective_finalize(iris_real *in_dest, collective_fft3D_state &fftstate)
{
    if(m_comm_len <= 0) {
	return;
    }

    int offset = 0;
    for(int i=0;i<m_comm_len;i++) {
	if(fftstate.recv_map[i] != -1) {
	    remap_item_gpu *plan = &m_recv_plans[fftstate.recv_map[i]];
	    plan->unpack(fftstate.recv_buff_gpu, offset, in_dest, plan->m_offset,fftstate.gpu_stream);
	    offset += plan->m_size;
	}
    }
}

