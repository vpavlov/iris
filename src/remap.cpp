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
#include "iris.h"
#include "remap.h"
#include "logger.h"
#include "memory.h"
#include "comm_rec.h"

using namespace ORG_NCSA_IRIS;

remap::remap(class iris *obj,
	     int *in_from_size, int *in_from_offset,
	     int *in_to_size, int *in_to_offset,
	     int in_unit_size,
	     int in_permute)
    : state_accessor(obj), m_send_plan(NULL), m_recv_plan(NULL), m_nsend(0),
      m_nrecv(0)
{
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

    // For each overlapping box, create a plan to send and store it in the
    // array of plans
    if(nsend) {
	m_send_plan = new remap_xfer_item_t[nsend];
	nsend = 0;
	int iproc = m_local_comm->m_rank;
	for(int i=0;i<m_local_comm->m_size;i++) {

	    iproc++;  // make sure self is last
	    if(iproc == m_local_comm->m_size) {
		iproc = 0;
	    }

	    box_t<int> overlap = m_from && to[iproc];
	    if(overlap.xsize > 0 && overlap.ysize > 0 && overlap.zsize > 0) {
		m_send_plan[nsend].peer = iproc;
		m_send_plan[nsend].offset = in_unit_size *
		    ((overlap.zlo - m_from.zlo) * m_from.ysize * m_from.xsize +
		     ((overlap.ylo - m_from.ylo) * m_from.xsize + 
		      overlap.xlo - m_from.xlo));
		m_send_plan[nsend].nx = in_unit_size * overlap.xsize;
		m_send_plan[nsend].ny = overlap.ysize;
		m_send_plan[nsend].nz = overlap.zsize;
		m_send_plan[nsend].stride_line = in_unit_size * m_from.xsize;
		m_send_plan[nsend].stride_plane = 
		    in_unit_size * m_from.xsize * m_from.ysize;
		m_send_plan[nsend].unit_size = in_unit_size;
		m_send_plan[nsend].size = in_unit_size *
		    overlap.xsize * overlap.ysize * overlap.zsize;
		m_send_plan[nsend].permute = 0;
		m_send_plan[nsend++].bufloc = 0;
	    }
	}

	// if we're sending to self, don't count it
	m_nsend = nsend;
	if(m_send_plan[nsend-1].peer == m_local_comm->m_rank) {
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
	m_recv_plan = new remap_xfer_item_t[nrecv];
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
		m_recv_plan[nrecv].peer = iproc;
		
		if(in_permute == 0) {
		    m_recv_plan[nrecv].offset = in_unit_size *
			((overlap.zlo - m_to.zlo) * m_to.ysize * m_to.xsize +
			 ((overlap.ylo - m_to.ylo) * m_to.xsize + 
			  overlap.xlo - m_to.xlo));
		    m_recv_plan[nrecv].nx = in_unit_size * overlap.xsize;
		    m_recv_plan[nrecv].ny = overlap.ysize;
		    m_recv_plan[nrecv].nz = overlap.zsize;
		    m_recv_plan[nrecv].stride_line = in_unit_size * m_to.xsize;
		    m_recv_plan[nrecv].stride_plane = 
			in_unit_size * m_to.xsize * m_to.ysize;
		    m_recv_plan[nrecv].unit_size = in_unit_size;
		}else if(in_permute == 1) {
		    m_recv_plan[nrecv].offset = in_unit_size *
			((overlap.xlo - m_to.xlo) * m_to.ysize * m_to.zsize +
			 ((overlap.zlo - m_to.zlo) * m_to.ysize + 
			  overlap.ylo - m_to.ylo));
		    m_recv_plan[nrecv].nx = overlap.xsize;
		    m_recv_plan[nrecv].ny = overlap.ysize;
		    m_recv_plan[nrecv].nz = overlap.zsize;
		    m_recv_plan[nrecv].stride_line = in_unit_size * m_to.ysize;
		    m_recv_plan[nrecv].stride_plane = 
			in_unit_size * m_to.ysize * m_to.zsize;
		    m_recv_plan[nrecv].unit_size = in_unit_size;
		}else {
		    m_recv_plan[nrecv].offset = in_unit_size *
			((overlap.ylo - m_to.ylo) * m_to.xsize * m_to.zsize +
			 ((overlap.xlo - m_to.xlo) * m_to.zsize + 
			  overlap.zlo - m_to.zlo));
		    m_recv_plan[nrecv].nx = overlap.xsize;
		    m_recv_plan[nrecv].ny = overlap.ysize;
		    m_recv_plan[nrecv].nz = overlap.zsize;
		    m_recv_plan[nrecv].stride_line = in_unit_size * m_to.zsize;
		    m_recv_plan[nrecv].stride_plane = 
			in_unit_size * m_to.zsize * m_to.xsize;
		    m_recv_plan[nrecv].unit_size = in_unit_size;
		}

		m_recv_plan[nrecv].size = in_unit_size *
		    overlap.xsize * overlap.ysize * overlap.zsize;
		m_recv_plan[nrecv].bufloc = bufloc;
		m_recv_plan[nrecv].permute = in_permute;
		bufloc += m_recv_plan[nrecv++].size;
	    }
	}

	// if we're recving from self, don't count it
	m_nrecv = nrecv;
	if(m_recv_plan[nrecv-1].peer == m_local_comm->m_rank) {
	    m_nrecv--;
	}

	m_self = true;
	if(nrecv == m_nrecv) {
	    m_self = false;
	}

    }

    int size = 0;
    for(int i=0;i<m_nsend;i++) {
	size = MAX(size, m_send_plan[i].size);
    }

    m_logger->trace("Remap will send do %d nodes", m_nsend);
    m_logger->trace("Remap will recv from %d nodes", m_nrecv);
    m_logger->trace("Remap largest message size is %d", size);

    if(size != 0) {
	m_sendbuf = (iris_real *)memory::wmalloc(size * sizeof(iris_real));
    }else {
	m_sendbuf = NULL;
    }

    memory::wfree(to);
    memory::wfree(from);
}

remap::~remap()
{
    if(m_send_plan != NULL) {
	delete [] m_send_plan;
    }

    if(m_recv_plan != NULL) {
	delete [] m_recv_plan;
    }

    memory::wfree(m_sendbuf);
}

void remap::perform(iris_real *in_src, iris_real *in_dest, iris_real *in_buf)
{
    MPI_Request *req = new MPI_Request[m_nsend];
    MPI_Win win;
    int *pending = m_iris->stos_fence_pending(&win);

    for(int i = 0; i < m_nsend; i++) {
	req[i] = MPI_REQUEST_NULL;
	remap_xfer_item_t *xfer = m_recv_plan + i;

	// flatten_brick(in_src + xfer->offset, in_buf, iris_real *dest,
	// 		   int nx, int ny, int nz, int stride_plane,
	// 		   int stride_line)
    }

    m_iris->stos_process_pending(pending, win);
    MPI_Waitall(m_nsend, req, MPI_STATUSES_IGNORE);
}
