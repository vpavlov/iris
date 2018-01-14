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
#include <unistd.h>
#include "comm_driver.h"
#include "memory.h"
#include "event.h"
#include "event_queue.h"

using namespace ORG_NCSA_IRIS;

static void *p2p_loop_start(void *thread_arg)
{
    comm_driver *obj = (comm_driver *)thread_arg;
    return obj->p2p_loop();
}

comm_driver::comm_driver(MPI_Comm in_comm, event_queue *in_queue)
    : m_comm(in_comm), m_queue(in_queue), m_quit(false),
      m_p2p_loop_thread_running(false)
{
}

comm_driver::~comm_driver()
{
    stop_event_source();
}

void comm_driver::start_event_source()
{
    if(!m_p2p_loop_thread_running) {
	pthread_create(&m_p2p_loop_thread, NULL, &p2p_loop_start, this);
	m_p2p_loop_thread_running = true;
    }
}

void comm_driver::stop_event_source()
{
    if(m_p2p_loop_thread_running) {
	void *retval;
	m_quit = true;
	pthread_join(m_p2p_loop_thread, &retval);
	m_p2p_loop_thread_running = false;
    }
}

void *comm_driver::p2p_loop()
{
    while(!m_quit) {
	MPI_Status status;
	int has_event;
	
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, m_comm, &has_event, &status);
	if(has_event) {
	    int nbytes;
	    MPI_Get_count(&status, MPI_BYTE, &nbytes);
	    void *msg = memory::wmalloc(nbytes);
	    MPI_Recv(msg, nbytes, MPI_BYTE,
		     status.MPI_SOURCE,
		     status.MPI_TAG,
		     m_comm, MPI_STATUS_IGNORE);
	    m_queue->post_event(m_comm, status.MPI_SOURCE,
				status.MPI_TAG, nbytes, msg);
	}

	// Without this, we sometimes get duplicate events. How is this even
	// possible? Is this a bug in MPI on the AVITOHOL (dev) machine? We
	// definitely have a single Send, but then we get two events. As if
	// the Recv is not really blocking and MPI_Iprobe returns has_event
	// = true second time for the same message?!?!?
	usleep(10);
    }
}
