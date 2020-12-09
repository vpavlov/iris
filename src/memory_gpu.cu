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

std::map<void *, std::map<std::string, void*> > memory_gpu::gpu_allocated_pointers;
std::map<void *, std::array<int,3> > memory_gpu::gpu_allocated_pointers_shape;

void *memory_gpu::wmalloc(int nbytes, void * parent,  const std::string label)
{
    void *retval = NULL;
    
    if (!label.empty()) {
        retval = get_registered_gpu_pointer(parent, label);
    }

    if (retval==NULL || label.empty()) {
           HANDLE_LAST_CUDA_ERROR;
    cudaError_t res = cudaMalloc((void**)&retval, nbytes);
    HANDLE_LAST_CUDA_ERROR;
    if(res != cudaSuccess) {
	throw std::bad_alloc();
    }
    if(!label.empty()) {
        register_gpu_pointer(parent,label,retval);
    }
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
    HANDLE_LAST_CUDA_ERROR;
	wfree(ptr);
	return tmp;
 };

void memory_gpu::wfree(void *ptr, bool keep_it)
{
    if ((!keep_it) && (ptr!=NULL)) {
    HANDLE_LAST_CUDA_ERROR;
    cudaFree(ptr);
    HANDLE_LAST_CUDA_ERROR;
    auto pl = get_parent_and_label(ptr);
    unregister_gpu_pointer(pl.first,pl.second);
    unregister_gpu_pointer_shape(ptr);
    }
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
    int threads = get_NThreads_1D(n);
    int blocks = get_NBlocks_X(n,threads);

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

__global__
void memory_set_kernel(iris_real** ptr2d, int n, iris_real val)
{
    iris_real *ptr = &(ptr2d[0][0]);
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

iris_real *memory_gpu::create_1d(iris_real *&array, int n1, bool clear,
                                    void * parent,  const std::string label)
{

    if ((!has_shape((void*)array,{n1,0,0}))&&(!label.empty())) {
        wfree(array);
    }

    array =  (iris_real *)wmalloc(sizeof(iris_real) * n1, parent, label);
    
    if(!label.empty()) {
    register_gpu_pointer_shape((void*)array,{n1,0,0});
    }

    if(clear) {
      int blocks = get_NBlocks_X(n1,IRIS_CUDA_SHARED_BLOCK_SIZE);
      int threads = IRIS_CUDA_SHARED_BLOCK_SIZE;
      memory_set_kernel<<<blocks,threads>>>(array,n1,(iris_real)0);
      cudaDeviceSynchronize();
      HANDLE_LAST_CUDA_ERROR;
    }
    return array;
};


void memory_gpu::destroy_1d(iris_real *&array, bool keep_it)
{
    if(array == NULL) {
	return;
	}

    if(keep_it) {
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

iris_real **memory_gpu::create_2d(iris_real **&array, int n1, int n2, bool clear, 
								void * parent,  const std::string label)
{
    if((!has_shape((void*)array,{n1,n2,0}))&&(!label.empty())) {
        destroy_2d(array);
    }

    int nitems = n1 * n2;

    void* ptr = get_registered_gpu_pointer(parent,label);
    
    if (ptr==NULL) {
        array =  (iris_real **)wmalloc(sizeof(iris_real *) * n1, parent, label);
        iris_real* data = (iris_real *)wmalloc(sizeof(iris_real) * nitems);
        int nthreads = get_NThreads_1D(n1);
        int nblocks = get_NBlocks_X(n1,nthreads);
        assign_2d_indexing_kernel<<<nblocks,nthreads>>>(array,data,n1,n2);
        register_gpu_pointer_shape(array,{n1,n2,0});
    } else {
        array = (iris_real **)ptr;
    }

    if(clear) {
        memory_set_kernel<<<get_NBlocks_X(nitems,IRIS_CUDA_SHARED_BLOCK_SIZE),IRIS_CUDA_SHARED_BLOCK_SIZE>>>(array,nitems,(iris_real)0);
    }

    
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;

    if(!label.empty()) {
    register_gpu_pointer_shape((void*)array,{n1,n2,0});
    }

    return array;
};

__global__
void get_2d_1d_pointer_kernel(iris_real **prt, iris_real *&ptr1d)
{
    ptr1d = prt[0];
}


void memory_gpu::destroy_2d(iris_real **&array, bool keep_it)
{
    if(array == NULL) {
    return;
    }

    if(!keep_it) {
    return;
    }

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
        //printf("xfrom %d xto %d array[%d] &tmp[%d]\n",xfrom,xto,i,m);
        }
        for (int j=yfrom; j<yto; ++j) {
            n = (m+j)*n3;
            tmp[m+j] = &data[n];
            //printf("tmp[%d] &data[%d] (i+1) %d j %d n3 %d\n",m+j,n,i+1,j,n3);
        }
    }
}


//**********************************************************************
// 3D Arrays
//**********************************************************************

iris_real ***memory_gpu::create_3d(iris_real ***&array, int n1, int n2, int n3,
                bool clear, iris_real init_val, void * parent,  const std::string label)
{
    int nitems = n1 * n2 * n3;

    if((!has_shape(array,{n1,n2,n3}))&&(!label.empty())) {
        destroy_3d(array);
    }

    void* ptr = get_registered_gpu_pointer(parent,label);
    if (ptr==NULL) {
        array   = (iris_real ***) wmalloc(sizeof(iris_real **) * n1,parent,label);
        iris_real **tmp = (iris_real **)  wmalloc(sizeof(iris_real *)  * n1 * n2);
        iris_real *data = (iris_real *)   wmalloc(sizeof(iris_real)    * nitems);
        int nthreads1 = get_NThreads_X(n1);
        int nthreads2 = get_NThreads_Y(n2);
        int nblocks1 = get_NBlocks_X(n1,nthreads1);
        int nblocks2 = get_NBlocks_Y(n2,nthreads2);

        assign_3d_indexing_kernel<<<dim3(nblocks1,nblocks2),dim3(nthreads1,nthreads2)>>>(array, tmp, data, n1, n2, n3);
        cudaDeviceSynchronize();
        HANDLE_LAST_CUDA_ERROR;
        if(!label.empty()) {
        register_gpu_pointer(parent,label,array);
        }
    } else {
        array = (iris_real***) ptr;
    }

    if(clear) {
      int blocks = get_NBlocks_X(nitems,IRIS_CUDA_SHARED_BLOCK_SIZE);
      int threads = IRIS_CUDA_SHARED_BLOCK_SIZE;
      memory_set_kernel<<<blocks,threads>>>(array,nitems, init_val);
    }
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;

    if(!label.empty()) {
    register_gpu_pointer_shape((void*)array,{n1,n2,0});
    }

    return array;
};


iris_real ***memory_gpu::create_3d(iris_real ***&array, int n1, int n2, int n3,
                bool clear, iris_real init_val, iris_real *&data, void * parent,  const std::string label)
{
    int nitems = n1 * n2 * n3;

    if((!has_shape(array,{n1,n2,n3}))&&(!label.empty())) {
        destroy_3d(array);
    }

    void* ptr = get_registered_gpu_pointer(parent,label);
    if (ptr==NULL) {
        array   = (iris_real ***) wmalloc(sizeof(iris_real **) * n1,parent,label);
        iris_real **tmp = (iris_real **)  wmalloc(sizeof(iris_real *)  * n1 * n2);
        data = (iris_real *)   wmalloc(sizeof(iris_real)    * nitems);
        int nthreads1 = get_NThreads_X(n1);
        int nthreads2 = get_NThreads_Y(n2);
        int nblocks1 = get_NBlocks_X(n1,nthreads1);
        int nblocks2 = get_NBlocks_Y(n2,nthreads2);

        assign_3d_indexing_kernel<<<dim3(nblocks1,nblocks2),dim3(nthreads1,nthreads2)>>>(array, tmp, data, n1, n2, n3);
        cudaDeviceSynchronize();
        HANDLE_LAST_CUDA_ERROR;
        if(!label.empty()) {
        register_gpu_pointer(parent,label,array);
        }
    } else {
        array = (iris_real***) ptr;
    }

    if(clear) {
      int blocks = get_NBlocks_X(nitems,IRIS_CUDA_SHARED_BLOCK_SIZE);
      int threads = IRIS_CUDA_SHARED_BLOCK_SIZE;
      memory_set_kernel<<<blocks,threads>>>(array,nitems, init_val);
    }
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;

    if(!label.empty()) {
    register_gpu_pointer_shape((void*)array,{n1,n2,0});
    }

    return array;
};

__global__
void get_3d_2d_1d_pointer_kernel(iris_real ***ptr3d,iris_real **&ptr2d, iris_real *&ptr1d)
{
    ptr2d = ptr3d[0];
    ptr1d = ptr3d[0][0];
}

void memory_gpu::destroy_3d(iris_real ***&array, bool keep_it)
{
  #warning "not sure if it really free the allocated mamory"
    if(array == NULL) {
    return;
    }

    if(keep_it) {
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

__global__
void get_data_pointer(iris_real ***array, iris_real **data_p)
{
    *data_p=&(array[0][0][0]);
}

int memory_gpu::sync_gpu_buffer(iris_real*** dst_gpu, void* src, size_t count)
{
    iris_real **gpu_data_p;
    iris_real *tmp;
    cudaMalloc(&gpu_data_p,sizeof(iris_real *));
    get_data_pointer<<<1,1>>>(dst_gpu,gpu_data_p);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    cudaMemcpy ( &tmp, gpu_data_p, sizeof(iris_real *), cudaMemcpyDeviceToHost);
    cudaFree(gpu_data_p);
    return sync_gpu_buffer(tmp,src,count);
}

int memory_gpu::sync_cpu_buffer(void* dst, iris_real*** src_gpu, size_t count)
{
	iris_real **gpu_data_p;
    iris_real *tmp;
    cudaMalloc(&gpu_data_p,sizeof(iris_real *));
    get_data_pointer<<<1,1>>>(src_gpu,gpu_data_p);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    cudaMemcpy ( &tmp, gpu_data_p, sizeof(iris_real *), cudaMemcpyDeviceToHost);
    cudaFree(gpu_data_p);
    return sync_cpu_buffer(dst,tmp,count);
}

int memory_gpu::sync_gpu_buffer(void* dst_gpu, const void* src, size_t count)
{
	return cudaMemcpy ( dst_gpu, src, count, cudaMemcpyHostToDevice);
}

int memory_gpu::sync_cpu_buffer(void* dst, const void* src_gpu, size_t count)
{
	return cudaMemcpy ( dst, src_gpu, count, cudaMemcpyDeviceToHost);
}

void * memory_gpu::get_registered_gpu_pointer(void *parent, std::string label)
{
    auto it = gpu_allocated_pointers.find(parent);
    if (it!=gpu_allocated_pointers.end()) {
        auto it1 = it->second.find(label);
        if (it1!=it->second.end()) {
            return it1->second;
        }
    }
    return NULL;
}

void memory_gpu::register_gpu_pointer(void *parent, std::string label, void* ptr)
{
    gpu_allocated_pointers[parent][label]=ptr;
}

void memory_gpu::unregister_gpu_pointer(void *parent, std::string label)
{
    auto it = gpu_allocated_pointers.find(parent);
    if (it!=gpu_allocated_pointers.end()) {
        auto it1 = it->second.find(label);
        if (it1!=it->second.end()) {
            it->second.erase(label);
        }
        if (it->second.empty())
        {
            gpu_allocated_pointers.erase(it);
        }
    }
}

bool memory_gpu::has_shape(void *ptr, std::array<int,3> in_shape)
{
    auto it = gpu_allocated_pointers_shape.find(ptr);
    if (it!=gpu_allocated_pointers_shape.end()) {
        if (it->second==in_shape) {
            return true;
        }
    }
    return false;
}

void memory_gpu::register_gpu_pointer_shape(void *ptr, std::array<int,3> in_shape)
{
    gpu_allocated_pointers_shape[ptr]=in_shape;
}

void memory_gpu::unregister_gpu_pointer_shape(void *ptr)
{
    auto it = gpu_allocated_pointers_shape.find(ptr);
    if (it!=gpu_allocated_pointers_shape.end()) {
        gpu_allocated_pointers_shape.erase(it);
    }
}

std::pair<void *,std::string> memory_gpu::get_parent_and_label(void* prt)
{
    for (auto it=gpu_allocated_pointers.begin(); it!=gpu_allocated_pointers.end();it++) {
            for (auto entry_it=it->second.begin(); entry_it!=it->second.end();entry_it++)
            {
                if (entry_it->second==prt)
                {
                    return std::pair<void*, std::string>(it->first,entry_it->first);
                }
            }
    }
    return std::pair<void*, std::string>(NULL,"");
}
