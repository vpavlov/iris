// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2019, the National Center for Supercomputing Applications
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
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include "timer.h"

using namespace ORG_NCSA_IRIS;

timer::timer()
{
    reset();
}

timer::~timer()
{
}

void timer::reset()
{
    m_cpu_accum = m_cpu_prev = m_wall_accum = m_wall_prev = 0.0;
}

void timer::start()
{
    m_cpu_prev = CPU_Time();
    m_wall_prev = MPI_Wtime();
}

void timer::stop()
{
    double cpu_curr = CPU_Time();
    double wall_curr = MPI_Wtime();

    m_cpu_accum += cpu_curr - m_cpu_prev;
    m_wall_accum += wall_curr - m_wall_prev;
}

inline double timer::CPU_Time()
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double retval = (double) ru.ru_utime.tv_sec;
    retval += (double) ru.ru_utime.tv_usec * 0.000001;
    return retval;
}
