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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_GPU_HALOEX_H__
#define __IRIS_GPU_HALOEX_H__

#include <mpi.h>
#include "real.h"

namespace ORG_NCSA_IRIS {

    class haloex_gpu {

    public:
	haloex_gpu(MPI_Comm in_comm, int *in_hood,
	       int in_mode,
	       iris_real ***in_data,
	       int *in_data_size,
	       int in_left_size,
	       int in_right_size,
	       int in_tag);

	~haloex_gpu();

	void exch_x() { exch(0); };
	void exch_y() { exch(1); };
	void exch_z() { exch(2); };
	void exch_full() { exch_x(); exch_y(); exch_z(); };

    private:

	void send(int in_dim, int in_dir);
	void recv(int in_dim, int in_dir);
	void exch(int in_dim);
	MPI_Comm m_comm;
	int *m_hood;
	int m_mode;
	iris_real ***m_data;
	int *m_data_size;
	int m_left_size;
	int m_right_size;
	int m_tag;

	iris_real *m_sendbufs[6];
	iris_real *m_recvbufs[6];

	MPI_Request m_req[6];
    };

}

#endif
