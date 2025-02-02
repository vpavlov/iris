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
#ifdef IRIS_CUDA
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>
#include "cuda.h"
#include "comm_rec.h"
#include "fmm.h"
#include "real.h"
#include "ses.h"
#include "fmm_kernels.h"
#include "fmm_pair.h"
#include "cuda_timer.h"

using namespace ORG_NCSA_IRIS;


//////////////
// Eval P2M //
//////////////

__device__ void madd_atomic(iris_real *M, int n, int m, complex<iris_real> &val)
{
    int c = n * (n + 1);
    atomicAdd(M+c-m, val.imag());
    atomicAdd(M+c+m, val.real());
}

IRIS_CUDA_DEVICE void ORG_NCSA_IRIS::d_p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_M)
{
    typedef cub::WarpReduce<complex<iris_real>> WarpReduce;
    
    __shared__ typename WarpReduce::TempStorage temp[32]; // warp size = 32
    
    iris_real r2 = x * x + y * y + z * z;

    complex<iris_real> R_m_m(q, 0);
    complex<iris_real> xy(x, y);

    int lane_id = threadIdx.x % 32;
    
    for(int m = 0; m < order; m++) {
	complex<iris_real> tt = WarpReduce(temp[lane_id]).Sum(R_m_m);
	if(lane_id == 0) {
	    madd_atomic(out_M, m, m, tt);
	}

	complex<iris_real> R_mplus1_m = z * R_m_m;
	tt = WarpReduce(temp[lane_id]).Sum(R_mplus1_m);
	if(lane_id == 0) {
	    madd_atomic(out_M, m+1, m, tt);
	}

	complex<iris_real> prev2 = R_m_m;
	complex<iris_real> prev1 = R_mplus1_m;
	for(int l = m+2; l <= order; l++) {
	    complex<iris_real> R_l_m = (2*l-1) * z * prev1 - r2 * prev2;
	    R_l_m /= (l * l - m * m);
	    tt = WarpReduce(temp[lane_id]).Sum(R_l_m);
	    if(lane_id == 0) {
		madd_atomic(out_M, l, m, tt);
	    }
	    prev2 = prev1;
	    prev1 = R_l_m;
	}

	R_m_m *= xy;
	R_m_m /= 2*(m+1);
    }
    complex<iris_real> tt = WarpReduce(temp[lane_id]).Sum(R_m_m);
    if(lane_id == 0) {
	madd_atomic(out_M, order, order, tt);
    }
}

__global__ void k_eval_p2m(cell_t *in_cells, int offset, particle_t *m_particles, int m_order, iris_real *m_M, int m_nterms)
{
    int leaf_idx = __fma(blockIdx.y, gridDim.z, blockIdx.z);   // Which leaf we are processing
    int cellID = leaf_idx + offset;                            // CellID of the leaf
    int j = IRIS_CUDA_TID;                                     // Target particle inside cellID
    
    cell_t *leaf = &in_cells[cellID];
    
    // no particles here -- continue
    if(j >= leaf->num_children) {
	return;
    }
    
    iris_real *M = m_M + cellID * m_nterms;

    iris_real x, y, z, q;
    x = m_particles[leaf->first_child+j].xyzq[0] - leaf->ses.c.r[0];
    y = m_particles[leaf->first_child+j].xyzq[1] - leaf->ses.c.r[1];
    z = m_particles[leaf->first_child+j].xyzq[2] - leaf->ses.c.r[2];
    q = m_particles[leaf->first_child+j].xyzq[3];
    d_p2m(m_order, x, y, z, q, M);
    
    leaf->flags |= IRIS_FMM_CELL_VALID_M;
}

void fmm::eval_p2m_gpu(cell_t *in_cells)
{
    // cuda_timer tm(m_streams[1]);

    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;

    dim3 nthreads(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks((m_max_particles-1)/IRIS_CUDA_NTHREADS + 1, nleafs, 1);
    
    // tm.start();
    k_eval_p2m<<<nblocks, nthreads, 0, m_streams[1]>>>(in_cells, offset, m_particles, m_order, m_M, m_nterms);
    // tm.stop();
    
    // m_logger->time("**** P2M time: %f ms (m_max_particles = %d, nleafs = %d)", tm.read(), m_max_particles, nleafs);
}

#endif
