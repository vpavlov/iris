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
#include "comm_driver.h"
#include "logger.h"

using namespace ORG_NCSA_IRIS;

comm_rec::comm_rec(iris *in_obj, MPI_Comm in_comm)
    : state_accessor(in_obj), m_comm(in_comm)
{
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);
    m_driver = new comm_driver(m_comm, m_queue);
}

comm_rec::~comm_rec()
{
    delete m_driver;
    // although we are not creating the communicator, we are freeing it.
    // This is done because iris::init allocates the communicators BEFORE
    // creating the drivers and event_queue to prevent the drivers from picking
    // up the event from the intercomm_create. Otherwise we could comm_dup the
    // comms in constructor and thus create them here.
    MPI_Comm_free(&m_comm);
}

MPI_Request comm_rec::post_event(void *in_data, int in_size,
				 int in_code, int in_peer)
{
    MPI_Request req;
    m_logger->trace("Posting event %x %d %d %d %p",
		    m_comm, in_peer, in_code, in_size, in_data);
    MPI_Isend(in_data, in_size, MPI_BYTE, in_peer, in_code, m_comm, &req);
    return req;
}

void comm_rec::send_event(void *in_data, int in_size, int in_code, int in_peer)
{
    MPI_Send(in_data, in_size, MPI_BYTE, in_peer, in_code, m_comm);
}

