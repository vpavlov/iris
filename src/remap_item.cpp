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
#include "remap_item.h"

using namespace ORG_NCSA_IRIS;

remap_item::remap_item()
{
}

remap_item::~remap_item()
{
}

void remap_item::pack(iris_real *src, iris_real *dest)
{
    int di = 0;
    for(int i = 0; i < m_nz; i++) {
	int plane = i * m_stride_plane;
	for(int j = 0; j < m_ny; j++) {
	    int si = plane + j * m_stride_line;
	    for(int k = 0; k < m_nx; k++) {
		dest[di++] = src[si++];
	    }
	}
    }
}

void remap_item::unpack(iris_real *src, iris_real *dest)
{
    int si = 0;
    for(int i = 0; i < m_nz; i++) {
	int plane = i * m_stride_plane;
	for(int j = 0; j < m_ny; j++) {
	    int di = plane + j * m_stride_line;
	    for(int k = 0; k < m_nx; k++) {
		dest[di++] = src[si++];
	    }
	}
    }
}
