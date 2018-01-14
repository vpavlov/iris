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
#ifndef __IRIS_EVENT_H__
#define __IRIS_EVENT_H__

#include <mpi.h>

namespace ORG_NCSA_IRIS {

#define IRIS_EVENT_QUIT              1
#define IRIS_EVENT_INTERCOMM_CREATE  2
#define IRIS_EVENT_LOCAL_BOXES       3

    struct event_t {
	MPI_Comm comm;// on which comm this occured
	int peer;     // MPI_SOURCE (in comm) / MPI_DEST
	int code;     // MPI_TAG
	int size;     // in bytes
	void *data;   // data allocated with memory::wmalloc

	event_t() = default;

	event_t(MPI_Comm in_comm, int in_peer, int in_code,
		int in_size, void *in_data) :
	    comm(in_comm), peer(in_peer), code(in_code), size(in_size),
	    data(in_data) {};
    };
    
}

#endif
