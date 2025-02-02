// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2021, the National Center for Supercomputing Applications
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
#include "cuda_timer.h"

using namespace ORG_NCSA_IRIS;

cuda_timer::cuda_timer(cudaStream_t in_stream)
    : m_stream(in_stream)
{
    cudaEventCreate(&m_start_event);
    cudaEventCreate(&m_stop_event);
    
    reset();
}

cuda_timer::~cuda_timer()
{
    cudaEventDestroy(m_start_event);
    cudaEventDestroy(m_stop_event);
}

void cuda_timer::start()
{
    cudaEventRecord(m_start_event, m_stream);
}

void cuda_timer::stop()
{
    float time;
    
    cudaEventRecord(m_stop_event, m_stream);
    cudaEventSynchronize(m_stop_event);
    cudaEventElapsedTime(&time, m_start_event, m_stop_event);

    m_accum += time;
}
