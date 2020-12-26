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
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <new>
#include <cstdint>
#include "memory.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

void *memory::wmalloc(size_t nbytes)
{
    void *retval;
    int res = posix_memalign(&retval, IRIS_MEMALIGN, nbytes);
    if(res != 0) {
	throw std::bad_alloc();
    }

    return retval;
}

void *memory::wrealloc(void *ptr, size_t nbytes)
{
    if(nbytes == 0) {
	wfree(ptr);
	return NULL;
    }

    ptr = realloc(ptr, nbytes);
    if(ptr == NULL) {
	throw std::bad_alloc();
    }

    if((uintptr_t)ptr % IRIS_MEMALIGN != 0) {
	void *tmp = wmalloc(nbytes);
	memcpy(tmp, ptr, MIN(nbytes, malloc_usable_size(ptr)));
	free(ptr);
	return tmp;
    }else {
	return ptr;
    }
}

void memory::wfree(void *ptr)
{
    free(ptr);
}

#define CPU_EXTRA_CAP 1.05

void *memory::wmalloc_cap(void *in_array, int in_new_size, int in_unit_size, int *io_capacity)
{
    if(in_array != NULL && in_new_size > *io_capacity) {
	memory::wfree(in_array);
    }

    if(in_array == NULL || in_new_size > *io_capacity) {
	*io_capacity = in_new_size * CPU_EXTRA_CAP;
	return memory::wmalloc(*io_capacity * in_unit_size);
    }
    return in_array;
}

void *memory::wrealloc_cap(void *in_array, int in_new_size, int in_unit_size, int *io_capacity)
{
    if(in_new_size > *io_capacity) {
	*io_capacity = in_new_size * CPU_EXTRA_CAP;
	return memory::wrealloc(in_array, *io_capacity * in_unit_size);
    }
    return in_array;
}
