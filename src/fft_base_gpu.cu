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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include "iris_gpu.h"
#include "fft_base_gpu.h"
#include "mesh_gpu.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

fft_base_gpu::fft_base_gpu(iris_gpu *obj, const char *in_name, bool in_use_collective)
    : state_accessor_gpu(obj), m_name(in_name), m_use_collective(in_use_collective),
      m_out_size { 0, 0, 0 }, m_out_offset { 0, 0, 0 },
      m_out_slow(-1), m_out_mid(-1), m_out_fast(-1),
      m_scratch(NULL)
{
#if defined _OPENMP
#if defined IRIS_CUDA
    FFTW_(init_threads);
    FFTW_(plan_with_nthreads(m_iris->m_nthreads));
#endif
#endif

    m_count = m_mesh->m_own_size[0] * m_mesh->m_own_size[1] * m_mesh->m_own_size[2];
    memory_gpu::create_1d(m_scratch, 2 * m_count);
}

fft_base_gpu::~fft_base_gpu()
{
    memory_gpu::destroy_1d(m_scratch);

#ifdef IRIS_CUDA
#ifdef _OPENMP
    FFTW_(cleanup_threads);
#else
    FFTW_(cleanup);
#endif
#endif
    
}
