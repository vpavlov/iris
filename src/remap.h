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
#ifndef __IRIS_REMAP_H__
#define __IRIS_REMAP_H__

#include "state_accessor.h"
#include "box.h"

namespace ORG_NCSA_IRIS {
    
    class remap : protected state_accessor {

    public:
	remap(class iris *obj,
	      int *in_from_offset, int *in_from_size, 
	      int *in_to_offset, int *in_to_size, 
	      int in_unit_size,
	      int in_permute);
	~remap();

	void perform(iris_real *in_src, iris_real *in_desc, iris_real *in_buf);

    private:
	box_t<int> m_from;
	box_t<int> m_to;
	int m_nsend;  // number of items to send
	int m_nrecv;  // number of items to recv
	class remap_item *m_send_plans;
	class remap_item *m_recv_plans;
	bool m_self;  // are we also receiving from self?
	iris_real *m_sendbuf;  // buffer for sending
    };

}

#endif
