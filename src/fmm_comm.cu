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
#include "cuda.h"
#include "fmm.h"

using namespace ORG_NCSA_IRIS;

int fmm::comm_LET_gpu()
{
    cudaMemcpyAsync(m_cells_cpu, m_cells, m_tree_size*sizeof(cell_t), cudaMemcpyDefault, m_streams[1]);
    cudaMemcpyAsync(m_M_cpu, m_M, m_tree_size*2*m_nterms*sizeof(iris_real), cudaMemcpyDefault, m_streams[1]);
    cudaStreamSynchronize(m_streams[1]);
    return comm_LET_cpu(m_cells_cpu, m_M_cpu);
}

__global__ void k_inhale_cells(unsigned char *m_recvbuf, int in_count, cell_t *m_xcells, iris_real *m_M, int unit_size, int m_nterms)
{
    int i = IRIS_CUDA_TID;
    if(i < in_count) {
	int cellID = *(int *)(m_recvbuf + unit_size * i);
	memcpy(&(m_xcells[cellID].ses), m_recvbuf + unit_size * i + sizeof(int), sizeof(sphere_t));
	memcpy(m_M + cellID*2*m_nterms, m_recvbuf + unit_size * i + sizeof(int) + sizeof(sphere_t), 2*m_nterms*sizeof(iris_real));
	m_xcells[cellID].flags |= (IRIS_FMM_CELL_ALIEN_NL | IRIS_FMM_CELL_VALID_M);
    }
}

void fmm::inhale_xcells_gpu(int in_count)
{
    int unit_size = sizeof(int) + sizeof(sphere_t) + 2*m_nterms*sizeof(iris_real);
    int rsize = in_count * unit_size;
    m_recvbuf_gpu = (unsigned char *)memory::wmalloc_gpu_cap(m_recvbuf_gpu, rsize, 1, &m_recvbuf_gpu_cap);
    cudaMemcpy(m_recvbuf_gpu, m_recvbuf, rsize, cudaMemcpyDefault);
    int nthreads = IRIS_CUDA_NTHREADS;
    int nblocks = IRIS_CUDA_NBLOCKS(in_count, nthreads);
    k_inhale_cells<<<nblocks, nthreads>>>(m_recvbuf_gpu, in_count, m_xcells, m_M, unit_size, m_nterms);
}
