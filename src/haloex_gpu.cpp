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
#include "haloex_gpu.h"
#include "memory.h"
#include <mpi.h>

using namespace ORG_NCSA_IRIS;

haloex_gpu::haloex_gpu(MPI_Comm in_comm, int *in_hood,
	       int in_mode,
	       iris_real ***in_data,
	       int *in_data_size,
	       int in_left_size,
	       int in_right_size,
	       int in_tag)
    : m_comm(in_comm), m_hood(in_hood), m_mode(in_mode), m_data(in_data),
      m_data_size(in_data_size), m_left_size(in_left_size),
      m_right_size(in_right_size), m_tag(in_tag)
{
	MPI_Comm_rank(m_comm,&m_rank);
    if(in_mode == 0) {
	m_sendbufs[0] = (iris_real *)  // buffer for sending to the right X
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));
	m_sendbufs[1] = (iris_real *)  // buffer for sending to the left X
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));

	m_sendbufs[2] = (iris_real *)  // buffer for sending to the top Y
	    memory_gpu::wmalloc(in_right_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));
	m_sendbufs[3] = (iris_real *)  // buffer for sending to the bottom Y
	    memory_gpu::wmalloc(in_left_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));
	
	m_sendbufs[4] = (iris_real *)  // buffer for sending to the front Z
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));
	m_sendbufs[5] = (iris_real *)  // buffer for sending to the back Z
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));


	m_recvbufs[0] = (iris_real *)  // buffer for recving from the left X
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));
	m_recvbufs[1] = (iris_real *)  // buffer for recving from the right X
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));

	m_recvbufs[2] = (iris_real *)  // buffer for recving from the bottom Y
	    memory_gpu::wmalloc(in_right_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));
	m_recvbufs[3] = (iris_real *)  // buffer for recving from the top Y
	    memory_gpu::wmalloc(in_left_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));
	
	m_recvbufs[4] = (iris_real *)  // buffer for recving from the back Z
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));
	m_recvbufs[5] = (iris_real *)  // buffer for recving from the front Z
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));
    }else {
	m_sendbufs[0] = (iris_real *)  // buffer for sending to the right X
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));
	m_sendbufs[1] = (iris_real *)  // buffer for sending to the left X
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));
	m_sendbufs[2] = (iris_real *)  // buffer for sending to the top Y
	    memory_gpu::wmalloc(in_left_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));
	m_sendbufs[3] = (iris_real *)  // buffer for sending to the bottom Y
	    memory_gpu::wmalloc(in_right_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));	
	m_sendbufs[4] = (iris_real *)  // buffer for sending to the front Z
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));
	m_sendbufs[5] = (iris_real *)  // buffer for sending to the back Z
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));

	m_recvbufs[0] = (iris_real *)  // buffer for recving from the left X
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));
	m_recvbufs[1] = (iris_real *)  // buffer for recving from the right X
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[2] *
			    sizeof(iris_real));
	m_recvbufs[2] = (iris_real *)  // buffer for recving from the bottom Y
	    memory_gpu::wmalloc(in_left_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));
	m_recvbufs[3] = (iris_real *)  // buffer for recving from the top Y
	    memory_gpu::wmalloc(in_right_size * m_data_size[0] * m_data_size[2] *
			    sizeof(iris_real));	
	m_recvbufs[4] = (iris_real *)  // buffer for recving from the back Z
	    memory_gpu::wmalloc(in_left_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));
	m_recvbufs[5] = (iris_real *)  // buffer for recving from the front Z
	    memory_gpu::wmalloc(in_right_size * m_data_size[1] * m_data_size[0] *
			    sizeof(iris_real));
    }
}
    
haloex_gpu::~haloex_gpu()
{
    for(int i=0;i<6;i++) {
	memory_gpu::wfree(m_sendbufs[i]);
	memory_gpu::wfree(m_recvbufs[i]);
    }
}

//
// Facilitate halo sending. This has two modes of operation:
//  - I calculate my own data + some extra layers. Then I send my extra
//    layers to my neighbor, which then adds them to his own data.
//  - I have my own data. Then I get some from my neighbour's data into some
//    extra layers and do some calculation on the totality;
//
// in_src is the 3D array that contains the array, including the extra items
// m_data_size[3] is an array with total size in each dimension
// in_mode = 0 to send outer layers; = 1 to send inner layers
// in_dim is the dimension to exchange (0 = X, 1 = Y, 2 = Z)
// in_dir is the direction of the exchange: 0 = "right", 1 = "left"
// in_tag is the first of a sequence of 6 tags that will be used for the exch
// out_sendbuf are sendbuffers which have to be kept until recv is made
// out_req are MPI_Requests to be waited upon
//
// This is how a line (let's say in X dimension) of in_src looks like (mode 0)
//
// |lll|ooooo|rrr|
//  \ / \   / \ /
//   A    B    C
// A = # of layers to send left;   = in_left_size
// B = # of layers of my own mesh; = in_own_size
// C = # of layers to send right;  = in_right_size
//
// This is how a line (let's say in X direction) of m_Ex_plus looks like:
//
// |000|eeeee|000|
//  \ / \   / \ /
//   A    B    C
// A = # of layers to receive from left;
// B = # of layers of my own mesh;
// C = # of layers to receive from right;
//
// TODO: optimization: handle the case when peer is self
void haloex_gpu::send(int in_dim, int in_dir)
{
    int A = m_left_size;
    int C = m_right_size;
    int B = m_data_size[in_dim] - (A+C);
        
    int ss, ns;
    
    if(m_mode == 0) {
	if(in_dir == 0) {
	    ss = A + B;
	    ns = C;
	}else {
	    ss = 0;
	    ns = A;
	}
    }else {
	if(in_dir == 0) {
	    ss = B;
	    ns = A;
	}else {
	    ss = A;
	    ns = C;
	}
    }

    int sx, nx, ex;
    int sy, ny, ey;
    int sz, nz, ez;
    
    if(in_dim == 0) {
	sx = ss;
	nx = ns;
	
	sy = 0;
	ny = m_data_size[1];

	sz = 0;
	nz = m_data_size[2];
    }else if(in_dim == 1) { 
	sx = 0;
	nx = m_data_size[0];

	sy = ss;
	ny = ns;

	sz = 0;
	nz = m_data_size[2];
    }else {  
	sx = 0;
	nx = m_data_size[0];

	sy = 0;
	ny = m_data_size[1];

	sz = ss;
	nz = ns;
    }

    ex = sx + nx;
    ey = sy + ny;
    ez = sz + nz;

    int idx = in_dim*2 + in_dir;
    iris_real *sendbuf = m_sendbufs[idx];
	int dest_rank = *(m_hood + idx);
    int size = nx*ny*nz*sizeof(iris_real);

	copy_to_sendbuf(sendbuf,m_data,sx,sy,sz,ex,ey,ez);
    
	if (dest_rank==m_rank)
	{
		size = 0;
	}

    MPI_Isend(sendbuf, size, MPI_BYTE, dest_rank,
	      m_tag + idx, m_comm, &m_req[idx]);
}

void haloex_gpu::recv(int in_dim, int in_dir)
{
    int A = m_left_size;
    int C = m_right_size;
    int B = m_data_size[in_dim] - (A+C);

    int ss, ns;
    
    if(m_mode == 0) {
	if(in_dir == 0) {
	    ss = A;
	    ns = C;
	}else {
	    ss = B;
	    ns = A;
	}
    }else {
	if(in_dir == 0) {
	    ss = 0;
	    ns = A;
	}else {
	    ss = A + B;
	    ns = C;
	}
    }

    int sx, nx, ex;
    int sy, ny, ey;
    int sz, nz, ez;
    
    if(in_dim == 0) {
	sx = ss;
	nx = ns;
	
	sy = 0;
	ny = m_data_size[1];

	sz = 0;
	nz = m_data_size[2];
    }else if(in_dim == 1) { 
	sx = 0;
	nx = m_data_size[0];

	sy = ss;
	ny = ns;

	sz = 0;
	nz = m_data_size[2];
    }else {
	sx = 0;
	nx = m_data_size[0];

	sy = 0;
	ny = m_data_size[1];

	sz = ss;
	nz = ns;
    }

    ex = sx + nx;
    ey = sy + ny;
    ez = sz + nz;

    int idx = in_dim*2 + in_dir;
    int size = nx*ny*nz*sizeof(iris_real);
	int src_rank = *(m_hood + in_dim*2 + 1 - in_dir);
	iris_real *recvbuf = m_recvbufs[idx];

	if (src_rank==m_rank)
	{
		recvbuf = m_sendbufs[idx];
		size = 0;
	}

    MPI_Recv(recvbuf, size, MPI_BYTE, src_rank,
	     m_tag + idx, m_comm, MPI_STATUS_IGNORE);

	copy_from_recvbuf(recvbuf, m_data, m_mode,
					  sx, sy, sz, ex, ey, ez);
}

void haloex_gpu::exch(int dim)
{
    send(dim, 0);
    send(dim, 1);
    recv(dim, 0);
    recv(dim, 1);
    MPI_Wait(&(m_req[dim*2]), MPI_STATUS_IGNORE);
    MPI_Wait(&(m_req[dim*2+1]), MPI_STATUS_IGNORE);
}

