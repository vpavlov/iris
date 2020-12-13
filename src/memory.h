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
#ifndef __IRIS_MEMORY_H__
#define __IRIS_MEMORY_H__

#include <stdlib.h>
#include <map>
#include <string>
#include <array>
#include <tuple>
#include "real.h"

namespace ORG_NCSA_IRIS {

    class memory {

    public:

	static void *wmalloc(size_t nbytes);
	static void *wrealloc(void *ptr, size_t nbytes);
	static void wfree(void *ptr);

	//**********************************************************************
	// 1D Arrays
	//**********************************************************************

	template <typename T>
	static T *create_1d(T *&array, int n1, bool clear = false)
	{
	    array =  (T *)wmalloc(sizeof(T) * n1);
	    if(clear) {
		for(int i=0;i<n1;i++) {
		    array[i] = (T)0;
		}
	    }
	    return array;
	}

	template <typename T>
	static void destroy_1d(T *&array)
	{
	    if(array == NULL) {
		return;
	    }

	    wfree(array);
	    array = NULL;
	}
	
	//**********************************************************************
	// 2D Arrays
	//**********************************************************************

	template <typename T>
	static T **create_2d(T **&array, int n1, int n2, bool clear = false)
	{
	    size_t nitems = n1 * n2;
	    array =  (T **)wmalloc(sizeof(T *) * n1);
	    T *data = (T *)wmalloc(sizeof(T) * nitems);
	    if(clear) {
		for(int i=0;i<nitems;i++) {
		    data[i] = (T)0;
		}
	    }

	    size_t n = 0;
	    for(int i = 0; i < n1; i++) {
		array[i] = &data[n];
		n += n2;
	    }

	    return array;
	}

	template <typename T>
	static void destroy_2d(T **&array)
	{
	    if(array == NULL) {
		return;
	    }

	    wfree(array[0]);  // free the data
	    wfree(array);     // free the array
	    array = NULL;
	}

	//**********************************************************************
	// 3D Arrays
	//**********************************************************************

	template <typename T>
	static T ***create_3d(T ***&array, int n1, int n2, int n3,
			      bool clear = false, T init_val = 0)
	{
	    size_t nitems = n1 * n2 * n3;
	    array   = (T ***) wmalloc(sizeof(T **) * n1);
	    T **tmp = (T **)  wmalloc(sizeof(T *)  * n1 * n2);
	    T *data = (T *)   wmalloc(sizeof(T)    * nitems);
	    if(clear) {
		for(int i=0;i<nitems;i++) {
		    data[i] = init_val;
		}
	    }

	    size_t n = 0, m;
	    for(int i = 0; i < n1; i++) {
		m = ((size_t) i) * n2;
		array[i] = &tmp[m];
		for(int j = 0; j < n2; j++) {
		    tmp[m+j] = &data[n];
		    n += n3;
		}
	    }

	    return array;
	}

	template <typename T>
	static void destroy_3d(T ***&array)
	{
	    if(array == NULL) {
		return;
	    }
	    
	    wfree(array[0][0]);
	    wfree(array[0]);
	    wfree(array);
	    array = NULL;
	}

	//**********************************************************************
	// 4D Arrays
	//**********************************************************************

	template <typename T>
	static T ****create_4d(T ****&array, int n1, int n2, int n3, int n4, 
			       bool clear = false)
	{
	    size_t nitems = n1 * n2 * n3 * n4;
	    array    = (T ****) wmalloc(sizeof(T ***) * n1);
	    T ***tmp = (T ***) wmalloc(sizeof(T **) * n1 * n2);
	    T **tmp2 = (T **)  wmalloc(sizeof(T *) * n1 * n2 * n3);
	    T *data  = (T *)   wmalloc(sizeof(T) * nitems);

	    if(clear) {
		for(int i=0;i<nitems;i++) {
		    data[i] = (T)0;
		}
	    }

	    size_t p = 0, m = 0, n = 0;
	    for(size_t k=0;k<n1;k++) {
		array[k] = &tmp[p];
		for(size_t j=0;j<n2;j++) {
		    tmp[j + p] = &tmp2[m];
		    for(size_t i=0;i<n3;i++) {
			tmp2[i + m] = &data[n];
			n += n4;
		    }
		    m += n3;
		}
		p += n2;
	    }

	    return array;
	}

	template <typename T>
	static void destroy_4d(T ****&array)
	{
	    if(array == NULL) {
		return;
	    }
	    
	    wfree(array[0][0][0]);
	    wfree(array[0][0]);
	    wfree(array[0]);
	    wfree(array);
	    array = NULL;
	}
    };
	
#ifdef IRIS_CUDA
#include <cuda_runtime_api.h>
    class memory_gpu {

    public:

	static void *wmalloc(int nbytes, void * parent=NULL, const std::string label="", bool host=false);
	static void *wrealloc(void *ptr, int nbytes, int old_size);
	static void wfree(void *ptr, bool keep_it=false, bool host=false);
	static int sync_gpu_buffer(void* dst_gpu, const void* src, size_t count);
	static int sync_cpu_buffer(void* dst, const void* src_gpu, size_t count);
	static int sync_gpu_buffer(iris_real*** dst_gpu, void* src, size_t count);
	static int sync_cpu_buffer(void* dst, iris_real*** src_gpu, size_t count);

	static int sync_gpu_buffer(void* dst_gpu, const void* src, size_t count, cudaStream_t &gpu_str);
	static int sync_cpu_buffer(void* dst, const void* src_gpu, size_t count, cudaStream_t &gpu_str);

	static int m_env_psp_cuda;

	static std::map<void *, std::map<std::string, void*> > gpu_allocated_pointers;
	static std::map<void *, std::array<int,3> > gpu_allocated_pointers_shape;

	//**********************************************************************
	// 1D Arrays
	//**********************************************************************
	static iris_real *create_1d(iris_real *&array, int n1, bool clear = false, 
								void * parent=NULL,  const std::string label="");
	
	static void destroy_1d(iris_real *&array, bool keep_it=false);

	//**********************************************************************
	// 2D Arrays
	//**********************************************************************
	static iris_real **create_2d(iris_real **&array, int n1, int n2, bool clear = false, 
								void * parent=NULL,  const std::string label="");
	
	static void destroy_2d(iris_real **&array, bool keep_it=false);

	//**********************************************************************
	// 3D Arrays
	//**********************************************************************
	static iris_real ***create_3d(iris_real ***&array, int n1, int n2, int n3,
			     				bool clear = false, iris_real init_val = 0, 
								void * parent=NULL,  const std::string label="");
	
	static iris_real ***create_3d(iris_real ***&array, int n1, int n2, int n3,
			     				bool clear, iris_real init_val, iris_real *&data,
								void * parent=NULL,  const std::string label="");

	static void destroy_3d(iris_real ***&array, bool keep_it=false);

	static void * get_registered_gpu_pointer(void *parent, std::string label);
	static void register_gpu_pointer(void *parent, std::string label, void *ptr);
	static void unregister_gpu_pointer(void *parent, std::string label);

	static bool has_shape(void *ptr, std::array<int,3> in_shape);
	static void register_gpu_pointer_shape(void *ptr, std::array<int,3> in_shape);
	static void unregister_gpu_pointer_shape(void *ptr);

	static std::pair<void *,std::string> get_parent_and_label(void* prt);
	};
	void sync_with_gpu();
#endif
};
iris_real calc_sum(iris_real ***v, int nx, int ny, int nz);
#endif
