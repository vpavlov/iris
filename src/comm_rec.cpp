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
#include <set>
#include <mpi.h>
#include "iris.h"
#include "comm_rec.h"
#include "logger.h"
#include "event.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

comm_rec::comm_rec(iris *in_obj, MPI_Comm in_comm)
    : state_accessor(in_obj), m_comm(in_comm)
{
    MPI_Comm_dup(in_comm, &m_comm);
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);
}

comm_rec::~comm_rec()
{
    MPI_Comm_free(&m_comm);
}

bool comm_rec::peek_event(event_t &out_event)
{
    int has_event;
    MPI_Status status;
    
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, m_comm, &has_event, &status);
    if(has_event) {
	out_event.comm = m_comm;
	out_event.peer = status.MPI_SOURCE;
	out_event.tag = status.MPI_TAG;
	MPI_Get_count(&status, MPI_BYTE, &(out_event.size));
	if(out_event.size != 0) {
	    out_event.data = memory::wmalloc(out_event.size);
	}else {
	    out_event.data = NULL;
	}
	MPI_Recv(out_event.data, out_event.size, MPI_BYTE,
		 out_event.peer, out_event.tag,
		 out_event.comm, MPI_STATUS_IGNORE);
	return true;
    }
    return false;
}

void comm_rec::get_event(event_t &out_event)
{
    get_event(MPI_ANY_SOURCE, MPI_ANY_TAG, out_event);
}

void comm_rec::get_event(int in_peer, int in_tag, event_t &out_event)
{
    MPI_Status status;
    
    MPI_Probe(in_peer, in_tag, m_comm, &status);
    out_event.comm = m_comm;
    out_event.peer = status.MPI_SOURCE;
    out_event.tag = status.MPI_TAG;
    MPI_Get_count(&status, MPI_BYTE, &(out_event.size));
    if(out_event.size != 0) {
	out_event.data = memory::wmalloc(out_event.size);
    }else {
	out_event.data = NULL;
    }
    MPI_Recv(out_event.data, out_event.size, MPI_BYTE,
	     out_event.peer, out_event.tag,
	     out_event.comm, MPI_STATUS_IGNORE);
}


