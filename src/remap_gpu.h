// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2019, the National Center for Supercomputing Applications
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
#ifndef __IRIS_GPU_REMAP_H__
#define __IRIS_GPU_REMAP_H__

#include "state_accessor_gpu.h"
#include "box.h"

namespace ORG_NCSA_IRIS {
    
    class remap_gpu : protected state_accessor_gpu {

    public:
	remap_gpu(class iris_gpu *obj,
	      int *in_from_offset, int *in_from_size, 
	      int *in_to_offset, int *in_to_size, 
	      int in_unit_size,
	      int in_permute, const char *in_name,
	      bool in_use_collective);
	~remap_gpu();

	void perform(iris_real *in_src, iris_real *in_dest, iris_real *in_buf)
	{
	    if(m_use_collective) {
		perform_collective(in_src, in_dest, in_buf);
	    }else {
		perform_p2p(in_src, in_dest, in_buf);
	    }
	}
	
    private:

	void perform_p2p(iris_real *in_src, iris_real *in_desc, iris_real *in_buf);
	void perform_collective(iris_real *in_src, iris_real *in_desc, iris_real *in_buf);
	
	char *m_name;
	box_t<int> m_from;
	box_t<int> m_to;
	int m_nsend;  // number of items to send
	int m_nrecv;  // number of items to recv
	class remap_item_gpu *m_send_plans;
	class remap_item_gpu *m_recv_plans;
	bool m_self;  // are we also receiving from self?
	iris_real *m_sendbuf;  // buffer for sending

	// collective support
	bool m_use_collective;
	int *m_comm_list;
	int m_comm_len;
	MPI_Comm m_collective_comm;
    };

}

#endif
