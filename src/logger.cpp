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
#include <cstdio>
#include <cstdlib>
#include <stdarg.h>
#include <string>
#include <omp.h>
#include "logger.h"
#include "comm_rec.h"

using namespace ORG_NCSA_IRIS;

#define DO_LOG(NUMERIC_LEVEL, TEXT_LEVEL)				\
    if(m_log_level >= NUMERIC_LEVEL) {					\
	char str[1024];							\
	double time = MPI_Wtime();					\
	va_list args;							\
	va_start(args, fmt);						\
	vsprintf(str, fmt, args);					\
	va_end(args);							\
	printf("[IRIS %d:%d:%0.6f] " TEXT_LEVEL ": %s\n",		\
	       m_uber_comm->m_rank, omp_get_thread_num(),		\
	       time - m_start_time, str);				\
    }

logger::logger(iris *obj)
    : state_accessor(obj)
{
    char *x;
    x = getenv("IRIS_LOG_LEVEL");
    if(x == NULL) {
	m_log_level = IRIS_LOG_LEVEL_ERROR;
    }else {
	std::string xx = x;
	if(xx == "error") {
	    m_log_level = IRIS_LOG_LEVEL_ERROR;
	}else if(xx == "warn") {
	    m_log_level = IRIS_LOG_LEVEL_WARN;
	}else if(xx == "info") {
	    m_log_level = IRIS_LOG_LEVEL_INFO;
	}else if(xx == "trace") {
	    m_log_level = IRIS_LOG_LEVEL_TRACE;
	}else {
	    fprintf(stderr, "Warning: unrecognized log level; will use 'error' level instead\n");
	}
    }

    m_start_time = MPI_Wtime();
}

logger::~logger()
{
}

void logger::trace(const char *fmt, ...)
{
    DO_LOG(IRIS_LOG_LEVEL_TRACE, "TRACE");
}

void logger::info(const char *fmt, ...)
{
    DO_LOG(IRIS_LOG_LEVEL_INFO, "INFO");
}

void logger::warn(const char *fmt, ...)
{
    DO_LOG(IRIS_LOG_LEVEL_WARN, "WARN");
}

void logger::error(const char *fmt, ...)
{
    DO_LOG(IRIS_LOG_LEVEL_ERROR, "ERROR");
}

void logger::trace_event(event_t *ev)
{
    trace("Event %x %d %d %d %p", ev->comm, ev->peer, ev->code, ev->size, ev->data);
}

void logger::trace_mem(void *in_ptr)
{
    unsigned char *ptr = (unsigned char *)in_ptr;
    trace("Memory @ %p starts with: 0x%02x%02x%02x%02x 0x%02x%02x%02x%02x", ptr,
	  ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}
