// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2020, the National Center for Supercomputing Applications
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
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <new>
#include <cstdint>
#include "memory.h"
#include "utils.h"
#include "memory_gpu.cuh"
#include "cuda_parameters.h"

using namespace ORG_NCSA_IRIS;

void *memory_gpu::wmalloc(size_t nbytes)
{
    void *retval;
    cudaError_t res = cudaMalloc(&retval, nbytes);
    if(res != cudaSuccess) {
	throw std::bad_alloc();
    }

    return retval;
};

void *memory_gpu::wrealloc(void *ptr, size_t nbytes, size_t old_size)
{
    if(nbytes == 0) {
	wfree(ptr);
	return NULL;
    }

	void *tmp = wmalloc(nbytes);
    cudaMemcpy(tmp, ptr, MIN(nbytes,old_size),cudaMemcpyDeviceToDevice);
	wfree(ptr);
	return tmp;
 };

void memory_gpu::wfree(void *ptr)
{
    cudaFree(ptr);
};


__global__
void memory_set_kernel(iris_real* ptr, size_t n, iris_real val)
{
    size_t ndx = IRIS_CUDA_INDEX(x);
    int chunk_size = IRIS_CUDA_CHUNK(x,n);
    size_t from = ndx*chunk_size;
    size_t to = MIN((ndx+1)*chunk_size,n);
    
    for(ndx=from; ndx!=to; ++ndx) {
    ptr[ndx]=val;
    }
};

//**********************************************************************
// 1D Arrays
//**********************************************************************

iris_real *memory_gpu::create_1d(iris_real *&array, int n1, bool clear)
{
    array =  (iris_real *)wmalloc(sizeof(iris_real) * n1);
    if(clear) {
        memory_set_kernel<<<get_NBlocks(n1,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(array,n1,(iris_real)0);
        cudaDeviceSynchronize();
    }
    return array;
};


void memory_gpu::destroy_1d(iris_real *&array)
{
    if(array == NULL) {
	return;
	}

	wfree(array);
	array = NULL;
};

////////////////////////////////////////////

__global__
void assign_2d_indexing_kernel(iris_real** array,iris_real* tmp, int n1, int n2)
{

    size_t xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,n1);
    size_t xfrom = xndx*xchunk_size;
    size_t xto = MIN((xndx+1)*xchunk_size,n1);

    for (xndx=xfrom; xndx!=xto; ++xndx) {
        int m = xndx*n2;
        array[xndx]=&tmp[m];
    }
};


//**********************************************************************
// 2D Arrays
//**********************************************************************

iris_real **memory_gpu::create_2d(iris_real **&array, int n1, int n2, bool clear)
{
    size_t nitems = n1 * n2;
    array =  (iris_real **)wmalloc(sizeof(iris_real *) * n1);
    iris_real* data = (iris_real *)wmalloc(sizeof(iris_real) * nitems);
    if(clear) {
        memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(data,nitems,(iris_real)0);
        cudaDeviceSynchronize();
    }

    assign_2d_indexing_kernel<<<get_NBlocks(n1,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(array,data,n1,n2);
    cudaDeviceSynchronize();

    return array;
};


void memory_gpu::destroy_2d(iris_real **&array)
{
    if(array == NULL) {
    return;
    }

    wfree(array[0]);  // free the data
    wfree(array);     // free the array
    array = NULL;
};

/////////////////////////////////////////////////////////

__global__
void assign_3d_indexing_kernel(iris_real*** array, iris_real** tmp, iris_real* data, int n1, int n2, int n3)
{
    size_t xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,n1);
    size_t yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,n2);

    size_t xfrom = xndx*xchunk_size;
    size_t xto = MIN((xndx+1)*xchunk_size,n1);

    size_t yfrom = yndx*ychunk_size;
    size_t yto = MIN((yndx+1)*ychunk_size,n2);

    for (xndx=xfrom; xndx!=xto; ++xndx) {
        size_t m = xndx*n2;
        array[xndx]=&tmp[m];
        for (yndx=yfrom; yndx!=yto; ++yndx) {
            size_t n = xndx*yndx*n3;
            tmp[xndx+yndx] = &data[n];
        }
    }
}


//**********************************************************************
// 3D Arrays
//**********************************************************************

iris_real ***memory_gpu::create_3d(iris_real ***&array, int n1, int n2, int n3,
bool clear, iris_real init_val)
{
    size_t nitems = n1 * n2 * n3;
    array   = (iris_real ***) wmalloc(sizeof(iris_real **) * n1);
    iris_real **tmp = (iris_real **)  wmalloc(sizeof(iris_real *)  * n1 * n2);
    iris_real *data = (iris_real *)   wmalloc(sizeof(iris_real)    * nitems);
    if(clear) {
        memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(data,nitems, init_val);
        cudaDeviceSynchronize();
    }

    int nblocks1 = get_NBlocks(n1,IRIS_CUDA_NTHREADS_2D);
    int nblocks2 = get_NBlocks(n2,IRIS_CUDA_NTHREADS_2D);
    int nthreads = IRIS_CUDA_NTHREADS_2D;
    
    assign_3d_indexing_kernel<<<dim3(nblocks1,nblocks2),dim3(nthreads,nthreads)>>>(array, tmp, data, n1, n2, n3);
    cudaDeviceSynchronize();
    return array;
};


void memory_gpu::destroy_3d(iris_real ***&array)
{
    if(array == NULL) {
    return;
    }
    
    wfree(array[0][0]);
    wfree(array[0]);
    wfree(array);
    array = NULL;
};
