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
#ifdef IRIS_CUDA
#include <stdexcept>
#include "memory.h"
#include "cuda.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

void *memory::wmalloc_gpu(int nbytes, bool clear, bool host)
{
    void *retval = NULL;
    cudaError_t res;
    if(host) {
	res = cudaMallocHost(&retval, nbytes);
    }else {
	res = cudaMalloc(&retval, nbytes);
    }
    if(res != cudaSuccess) {
	printf("CUDA Error: %s - %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
	throw std::runtime_error("CUDA Exception occured");
    }

    if(clear) {
	cudaMemset(retval, 0, nbytes);
    }
    
    return retval;
}

void memory::wfree_gpu(void *ptr, bool host)
{
    cudaError_t res;
    if(host) {
	res = cudaFreeHost(ptr);
    }else {
	res = cudaFree(ptr);
    }
    if(res != cudaSuccess) {
	printf("CUDA Error: %s - %s\n", cudaGetErrorName(res), cudaGetErrorString(res));
	throw std::runtime_error("CUDA Exception occured");
    }
}
    
void memory::create_1d_gpu(iris_real *&array, int n, bool clear)
{
    array = (iris_real *)wmalloc_gpu(sizeof(iris_real) * n, clear);
}

void memory::destroy_1d_gpu(iris_real *&array)
{
    if(array == NULL) {
	return;
    }
    wfree_gpu(array);
    array = NULL;
}

__global__ void k_create_2d_helper(iris_real **array, iris_real *data, int n1, int n2)
{
    IRIS_CUDA_SETUP_WS(n1);
    for(int i=from;i<to;i++) {
	array[i] = data + i * n2;
    }
}

iris_real **memory::create_2d_gpu(iris_real **&array, int n1, int n2, bool clear)
{
    int nitems = n1 * n2;
    array = (iris_real **)wmalloc_gpu(n1 * sizeof(iris_real *));
    iris_real *data = (iris_real *)wmalloc_gpu(nitems * sizeof(iris_real), clear);
    int nthreads = IRIS_CUDA_NTHREADS;
    int nblocks IRIS_CUDA_NBLOCKS(nitems, nthreads);
    k_create_2d_helper<<<nblocks, nthreads>>>(array, data, n1, n2);
    return array;
}

__global__ void k_get_data_ptr(iris_real **array, iris_real *&data)
{
    data = array[0];
}

void memory::destroy_2d_gpu(iris_real **&array)
{
    if(array == NULL) {
	return;
    }
    
    iris_real *data;
    k_get_data_ptr<<<1, 1>>>(array, data);
    cudaDeviceSynchronize();
    wfree_gpu(data);
    wfree_gpu(array);
    array = NULL;
}

#define GPU_EXTRA_CAP 1.05

void *memory::wmalloc_gpu_cap(void *in_array, int in_new_size, int in_unit_size, int *io_capacity, bool host)
{
    if(in_array != NULL && in_new_size > *io_capacity) {
	memory::wfree_gpu(in_array, host);
    }

    if(in_array == NULL || in_new_size > *io_capacity) {
	*io_capacity = in_new_size * GPU_EXTRA_CAP;
	return memory::wmalloc_gpu(*io_capacity * in_unit_size, false, host);
    }
    return in_array;
}

#endif
