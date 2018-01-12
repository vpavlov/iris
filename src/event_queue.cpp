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
#include "event_queue.h"

using namespace ORG_NCSA_IRIS;

event_queue::event_queue(iris *obj)
    :state_accessor(obj)
{
    pthread_mutex_init(&m_qmutex, NULL);
    pthread_cond_init(&m_qcond, NULL);
}

event_queue::~event_queue()
{
    pthread_cond_destroy(&m_qcond);
    pthread_mutex_destroy(&m_qmutex);
}

void event_queue::post_event(MPI_Comm comm, int peer, int code,
			     int size, void *data)
{
    pthread_mutex_lock(&m_qmutex);
    m_queue.emplace_back(comm, peer, code, size, data);
    pthread_cond_signal(&m_qcond);
    pthread_mutex_unlock(&m_qmutex);
}

void event_queue::post_quit_event_self()
{
    post_event(0, 0, IRIS_EVENT_QUIT, 0, NULL);
}


bool event_queue::get_event(event_t &event)
{
    pthread_mutex_lock(&m_qmutex);
    while(m_queue.empty()) {
	pthread_cond_wait(&m_qcond, &m_qmutex);
    }

    event = m_queue.front();
    m_queue.pop_front();
    pthread_mutex_unlock(&m_qmutex);
    return event.code != IRIS_EVENT_QUIT;
}
