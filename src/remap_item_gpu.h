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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_GPU_REMAP_ITEM_H__
#define __IRIS_GPU_REMAP_ITEM_H__

#include "real.h"

namespace ORG_NCSA_IRIS {

    class remap_item_gpu {

    public:
	remap_item_gpu();
	~remap_item_gpu();

	void pack(iris_real ***src, int src_offset, iris_real *dest, int dest_offset);
	void pack(iris_real *src, int src_offset, iris_real *dest, int dest_offset);
	virtual void unpack(iris_real *src, int src_offset, iris_real *dest, int dest_offset);

    public:
	int m_peer;            // processor to send to/recv from
	int m_offset;          // offset from [0, 0, 0] of the start of block
	int m_nx, m_ny, m_nz;  // number of items in X, Y, Z that have to be xf
	int m_stride_line;     // distance to next pencil
	int m_stride_plane;    // distance to next plane
	int m_size;            // size of the data to send
	int m_bufloc;          // only in recv: offset in receiving buffer
    };

};

#endif
