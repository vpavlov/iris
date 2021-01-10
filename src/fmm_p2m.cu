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
#include "cuda.h"
#include "comm_rec.h"
#include "fmm.h"
#include "real.h"
#include "ses.h"
#include "fmm_kernels.h"
#include "fmm_pair.h"

using namespace ORG_NCSA_IRIS;


//////////////
// Eval P2M //
//////////////

__device__ void multipole_atomic_add(iris_real *M, int l, int m, complex<iris_real> &val)
{
    int i = multipole_index(l, m);
    atomicAdd(M+i, val.real());
    atomicAdd(M+i+1, val.imag());
}

__device__ void d_p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_M)
{
    iris_real r2 = x * x + y * y + z * z;

    complex<iris_real> R_m_m(q, 0);
    complex<iris_real> xy(x, y);
    
    for(int m = 0; m < order; m++) {
	multipole_atomic_add(out_M, m, m, R_m_m);

	complex<iris_real> R_mplus1_m = z * R_m_m;
	multipole_atomic_add(out_M, m+1, m, R_mplus1_m);

	complex<iris_real> prev2 = R_m_m;
	complex<iris_real> prev1 = R_mplus1_m;
	for(int l = m+2; l <= order; l++) {
	    complex<iris_real> R_l_m = (2*l-1) * z * prev1 - r2 * prev2;
	    R_l_m /= (l * l - m * m);
	    multipole_atomic_add(out_M, l, m, R_l_m);
	    prev2 = prev1;
	    prev1 = R_l_m;
	}

	R_m_m *= xy;
	R_m_m /= 2*(m+1);
    }
    multipole_atomic_add(out_M, order, order, R_m_m);
}


__global__ void k_eval_p2m(cell_t *in_cells, int offset, bool alien_only, particle_t *m_particles, particle_t *m_xparticles, int m_order, iris_real *m_M, int m_nterms)
{
    int leaf_idx = blockIdx.y * gridDim.z + blockIdx.z;   // Which interaction pair we're processing
    int cellID = leaf_idx + offset;
    int j = IRIS_CUDA_TID;                                // Target particle inside cellID
    
    cell_t *leaf = &in_cells[cellID];
    
    // no particles here -- continue
    if(j >= leaf->num_children) {
	return;
    }
    
    // we only want alien cells, but this one is local -- continue
    if(alien_only && !(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF)) {
	return;
    }
    
    // it has been send from exchange_LET AND from halo exchange -- continue
    if(alien_only && (leaf->flags & IRIS_FMM_CELL_ALIEN_NL)) {
	return;
    }
    
    iris_real *M = m_M + cellID * 2 * m_nterms;

    iris_real x, y, z, q;
    if(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF) {
	x = m_xparticles[leaf->first_child+j].xyzq[0] - leaf->ses.c.r[0];
	y = m_xparticles[leaf->first_child+j].xyzq[1] - leaf->ses.c.r[1];
	z = m_xparticles[leaf->first_child+j].xyzq[2] - leaf->ses.c.r[2];
	q = m_xparticles[leaf->first_child+j].xyzq[3];
    }else {
	x = m_particles[leaf->first_child+j].xyzq[0] - leaf->ses.c.r[0];
	y = m_particles[leaf->first_child+j].xyzq[1] - leaf->ses.c.r[1];
	z = m_particles[leaf->first_child+j].xyzq[2] - leaf->ses.c.r[2];
	q = m_particles[leaf->first_child+j].xyzq[3];
    }
    
    d_p2m(m_order, x, y, z, q, M);
    
    leaf->flags |= IRIS_FMM_CELL_VALID_M;
}

void fmm::eval_p2m_gpu(cell_t *in_cells, bool alien_only)
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;

    dim3 nthreads(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks((m_max_particles-1)/IRIS_CUDA_NTHREADS + 1, nleafs, 1);
    k_eval_p2m<<<nblocks, nthreads>>>(in_cells, offset, alien_only, m_particles, m_xparticles, m_order, m_M, m_nterms);
}

#endif
