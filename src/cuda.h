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
#ifndef __IRIS_CUDA_H__
#define __IRIS_CUDA_H__

#ifdef IRIS_CUDA

#define IRIS_CUDA_NTHREADS 256
#define IRIS_CUDA_NBLOCKS(N, NT) (((N) + (NT) - 1)/(NT))
#define IRIS_CUDA_TID (blockIdx.x * blockDim.x + threadIdx.x)
#define IRIS_CUDA_CHUNK(N) (((N) + gridDim.x*blockDim.x - 1)/(gridDim.x*blockDim.x))

#define IRIS_CUDA_SETUP_WS(N)			\
    int tid = IRIS_CUDA_TID;			\
    int chunk_size = IRIS_CUDA_CHUNK((N));	\
    int from = tid * chunk_size;		\
    int to = from + chunk_size;			\
    to = MIN(to, (N))

#define IRIS_CUDA_HANDLE_ERROR(res)					\
    if(res != cudaSuccess) {						\
	m_logger->error("CUDA Error: %s - %s", cudaGetErrorName(res), cudaGetErrorString(res)); \
	throw std::runtime_error("CUDA Exception occured");		\
    }

#define IRIS_CUDA_CHECK_ERROR {						\
    cudaError_t err = cudaGetLastError();				\
    if(err != cudaSuccess) {						\
	m_logger->error("CUDA Error: %s - %s", cudaGetErrorName(err), cudaGetErrorString(err)); \
	throw std::runtime_error("CUDA Exception occured");		\
    }									\
    }

#undef IRIS_CUDA_DEVICE_HOST
#define IRIS_CUDA_DEVICE_HOST __device__ __host__

#else

#undef IRIS_CUDA_DEVICE_HOST
#define IRIS_CUDA_DEVICE_HOST

#endif

#endif
