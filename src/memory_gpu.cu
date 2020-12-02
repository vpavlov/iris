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

void *memory_gpu::wmalloc(int nbytes)
{
    void *retval;
    cudaError_t res = cudaMalloc(&retval, nbytes);
    if(res != cudaSuccess) {
	throw std::bad_alloc();
    }

    return retval;
};

void *memory_gpu::wrealloc(void *ptr, int nbytes, int old_size)
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
void memory_set_kernel(iris_real* ptr, int n, iris_real val)
{
    int ndx = IRIS_CUDA_INDEX(x);
    int chunk_size = IRIS_CUDA_CHUNK(x,n);
    int from = ndx*chunk_size;
    int to = MIN((ndx+1)*chunk_size,n);
    
    for(ndx=from; ndx<to; ++ndx) {
    ptr[ndx]=val;
    }
};

__global__
void print_kernel(iris_real* ptr, int n, const char* name)
{
    int ndx = IRIS_CUDA_INDEX(x);
    int chunk_size = IRIS_CUDA_CHUNK(x,n);
    int from = ndx*chunk_size;
    int to = MIN((ndx+1)*chunk_size,n);
    
    for(ndx=from; ndx<to; ++ndx) {
        if(ndx==0)
    printf("%s[%d] %f",name,ndx,ptr[ndx]);
    }
};

void print_vector_gpu(iris_real* ptr, int n, const char* name)
{   
    int blocks = get_NBlocks(n,IRIS_CUDA_NTHREADS);
    int threads = MIN((n+blocks+1)/blocks,IRIS_CUDA_NTHREADS);
    print_kernel<<<blocks,threads>>>(ptr,n,name);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
};


__global__
void memory_set_kernel(iris_real*** ptr3d, int n, iris_real val)
{
    iris_real *ptr = &(ptr3d[0][0][0]);
    int ndx = IRIS_CUDA_INDEX(x);
    int chunk_size = IRIS_CUDA_CHUNK(x,n);
    int from = ndx*chunk_size;
    int to = MIN((ndx+1)*chunk_size,n);
    
    for(ndx=from; ndx<to; ++ndx) {
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
      int blocks = get_NBlocks(n1,IRIS_CUDA_NTHREADS);
      int threads = MIN((n1+blocks+1)/blocks,IRIS_CUDA_NTHREADS);
      memory_set_kernel<<<blocks,threads>>>(array,n1,(iris_real)0);
      cudaDeviceSynchronize();
      HANDLE_LAST_CUDA_ERROR;
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

    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,n1);
    int xfrom = xndx*xchunk_size;
    int xto = MIN((xndx+1)*xchunk_size,n1);

    for (xndx=xfrom; xndx<xto; ++xndx) {
        int m = xndx*n2;
        array[xndx]=&tmp[m];
    }
};


//**********************************************************************
// 2D Arrays
//**********************************************************************

iris_real **memory_gpu::create_2d(iris_real **&array, int n1, int n2, bool clear)
{
    int nitems = n1 * n2;
    array =  (iris_real **)wmalloc(sizeof(iris_real *) * n1);
    iris_real* data = (iris_real *)wmalloc(sizeof(iris_real) * nitems);
    if(clear) {
        memory_set_kernel<<<get_NBlocks(nitems,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(data,nitems,(iris_real)0);
        HANDLE_LAST_CUDA_ERROR;
    }

    assign_2d_indexing_kernel<<<get_NBlocks(n1,IRIS_CUDA_NTHREADS),IRIS_CUDA_NTHREADS>>>(array,data,n1,n2);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;

    return array;
};

__global__
void get_2d_1d_pointer_kernel(iris_real **prt, iris_real *&ptr1d)
{
    ptr1d = prt[0];
}


void memory_gpu::destroy_2d(iris_real **&array)
{
    if(array == NULL) {
    return;
    }

    //    wfree(array[0]);  // free the data
    iris_real *data;
    get_2d_1d_pointer_kernel<<<1,1>>>(array,data);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    wfree(data);
    wfree(array);     // free the array
    array = NULL;
};

/////////////////////////////////////////////////////////

__global__
void assign_3d_indexing_kernel(iris_real*** array, iris_real** tmp, iris_real* data, int n1, int n2, int n3)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,n1);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,n2);

    int xfrom = xndx*xchunk_size;
    int xto = MIN((xndx+1)*xchunk_size,n1);

    int yfrom = yndx*ychunk_size;
    int yto = MIN((yndx+1)*ychunk_size,n2);
    int m,n;
    for (int i=xfrom; i<xto; ++i) {
        m = i*n2;
        if (yfrom==0){
        array[i]=&tmp[m];
     //   printf("xfrom %d xto %d array[%d] &tmp[%d]\n",xfrom,xto,i,m);
        }
        for (int j=yfrom; j<yto; ++j) {
            n = (m+j)*n3;
            tmp[m+j] = &data[n];
           // printf("tmp[%d] &data[%d] (i+1) %d j %d n3 %d\n",m+j,n,i+1,j,n3);
        }
    }
}


//**********************************************************************
// 3D Arrays
//**********************************************************************

iris_real ***memory_gpu::create_3d(iris_real ***&array, int n1, int n2, int n3,
bool clear, iris_real init_val)
{
    int nitems = n1 * n2 * n3;
    array   = (iris_real ***) wmalloc(sizeof(iris_real **) * n1);
    iris_real **tmp = (iris_real **)  wmalloc(sizeof(iris_real *)  * n1 * n2);
    iris_real *data = (iris_real *)   wmalloc(sizeof(iris_real)    * nitems);
    if(clear) {
      int blocks = get_NBlocks(nitems,IRIS_CUDA_NTHREADS);
      int threads = MIN((nitems+blocks+1)/blocks,IRIS_CUDA_NTHREADS);
      memory_set_kernel<<<blocks,threads>>>(data,nitems, init_val);
        HANDLE_LAST_CUDA_ERROR;
    }

    int nblocks1 = get_NBlocks(n1,IRIS_CUDA_NTHREADS_2D);
    int nblocks2 = get_NBlocks(n2,IRIS_CUDA_NTHREADS_2D);
    int nthreads1 = MIN((n1+nblocks1+1)/nblocks1,IRIS_CUDA_NTHREADS_2D);
    int nthreads2 = MIN((n2+nblocks2+1)/nblocks2,IRIS_CUDA_NTHREADS_2D);
    assign_3d_indexing_kernel<<<dim3(nblocks1,nblocks2),dim3(nthreads1,nthreads2)>>>(array, tmp, data, n1, n2, n3);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    return array;
};

__global__
void get_3d_2d_1d_pointer_kernel(iris_real ***ptr3d,iris_real **&ptr2d, iris_real *&ptr1d)
{
    ptr2d = ptr3d[0];
    ptr1d = ptr3d[0][0];
}

void memory_gpu::destroy_3d(iris_real ***&array)
{
  #warning "not sure if it really free the allocated mamory"
    if(array == NULL) {
    return;
    }
    //size_t free, total;

    //printf("cudaMemGetInfo LAST CUDA EROOR: %s\n",cudaGetErrorString ( cudaGetLastError()  ));
    //wfree((void*)&array[0][0][0]);
    //printf("(void*)&array[0][0][0] LAST CUDA EROOR: %s\n",cudaGetErrorString ( cudaGetLastError()  ));
    
    iris_real **tmpmap;
    iris_real *datap;
    get_3d_2d_1d_pointer_kernel<<<1,1>>>(array,tmpmap,datap);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;

   wfree(datap);
   wfree(tmpmap);
   wfree(array);
    
    //printf("array LAST CUDA EROOR: %s\n",cudaGetErrorString ( cudaGetLastError()  ));
    //cudaMemGetInfo(&free,&total);
    //printf("free %d total %d\n");

    array = NULL;
};

int memory_gpu::sync_gpu_buffer(void* dst_gpu, const void* src, size_t count)
{
	return cudaMemcpy ( dst_gpu, src, count, cudaMemcpyHostToDevice);
}

int memory_gpu::sync_cpu_buffer(void* dst, const void* src_gpu, size_t count)
{
	return cudaMemcpy ( dst, src_gpu, count, cudaMemcpyDeviceToHost);
}
