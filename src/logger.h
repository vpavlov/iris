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
#ifndef __IRIS_LOGGER_H__
#define __IRIS_LOGGER_H__

#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

#define IRIS_LOG_LEVEL_ERROR  0
#define IRIS_LOG_LEVEL_WARN   1
#define IRIS_LOG_LEVEL_INFO   2
#define IRIS_LOG_LEVEL_TIME   3
#define IRIS_LOG_LEVEL_TRACE  4
    
    class logger : protected state_accessor {

    public:

	logger(class iris *obj);
	~logger();

	void set_log_level(int in_log_level) { m_log_level = in_log_level; };

	void trace(const char *str, ...);
	void time(const char *str, ...);
	void info(const char *str, ...);
	void warn(const char *str, ...);
	void error(const char *str, ...);

	void trace_mem(void *addr);
	void trace_event(event_t *in_event);

    private:
	double m_log_level;
	double m_start_time;
    };

}

#endif
