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
#ifndef __IRIS_FMM_GPU_HALLOC_H__
#define __IRIS_FMM_GPU_HALLOC_H__

#include <limits>

namespace ORG_NCSA_IRIS {

    template<class T>
    class HostAlloc {
	
    public:
	// inner types
	typedef T           *pointer;
	typedef const T     *const_pointer;
	typedef void        *void_pointer;
	typedef const void  *const_void_pointer;
	typedef T            value_type;
	typedef size_t       size_type;
	typedef ptrdiff_t    difference_type;
	typedef T           &reference;
	typedef const T     &const_reference;

	template <class U>
	struct rebind {
	    typedef HostAlloc<U> other;
	};

	pointer allocate(size_type num, const void * = 0)
	{
	    void *ptr;
	    cudaMallocHost(&ptr, num * sizeof(T));
	    return (pointer)ptr;
	}

	void deallocate (pointer p, size_type num)
	{
	    cudaFreeHost(p);
	}

	// return maximum number of elements that can be allocated
	size_type max_size () const throw()
	{
	    return std::numeric_limits<size_t>::max() / sizeof(T);
	}

	// initialize elements of allocated storage p with value value
	void construct (pointer p, const T& value)
	{
	    new((void *)p)T(value);
	}

	// destroy elements of initialized storage p
	void destroy (pointer p)
	{
	    p->~T();
	}
	

	// return address of values
	pointer address (reference value) const
	{
	    return &value;
	}
	
	const_pointer address (const_reference value) const
	{
	    return &value;
	}

	HostAlloc() throw() {}
	HostAlloc(const HostAlloc&) throw() {}
	template <class U> HostAlloc (const HostAlloc<U>&) throw() {}
	~HostAlloc() throw() {}
    };
}

#endif
