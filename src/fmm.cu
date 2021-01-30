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
#include <thrust/device_vector.h>
#include "cuda.h"
#include "comm_rec.h"
#include "fmm.h"
#include "real.h"
#include "ses.h"
#include "fmm_kernels.h"
#include "fmm_pair.h"

using namespace ORG_NCSA_IRIS;


////////////////////
// Load Particles //
////////////////////


__global__ void k_load_charges(iris_real *charges, int ncharges, int hwm,
			       iris_real xlo, iris_real ylo, iris_real zlo,
			       iris_real lsx, iris_real lsy, iris_real lsz,
			       int max_level, int offset, particle_t *m_particles, int rank)
{
    IRIS_CUDA_SETUP_WS(ncharges);
    for(int i=from;i<to;i++) {
	iris_real tx = (charges[i * 5 + 0] - xlo) / lsx;
	iris_real ty = (charges[i * 5 + 1] - ylo) / lsy;
	iris_real tz = (charges[i * 5 + 2] - zlo) / lsz;
	
	int cellID = cell_meta_t::leaf_coords_to_ID(tx, ty, tz, max_level);
	int chargeID = (int)charges[i*5 + 4];
	
	m_particles[i+hwm].rank = rank;
	m_particles[i+hwm].index = chargeID;
	m_particles[i+hwm].cellID = cellID;
	m_particles[i+hwm].xyzq[0] = charges[i*5+0];
	m_particles[i+hwm].xyzq[1] = charges[i*5+1];
	m_particles[i+hwm].xyzq[2] = charges[i*5+2];
	m_particles[i+hwm].xyzq[3] = charges[i*5+3];
	m_particles[i+hwm].tgt[0] = 0.0;
	m_particles[i+hwm].tgt[1] = 0.0;
	m_particles[i+hwm].tgt[2] = 0.0;
	m_particles[i+hwm].tgt[3] = 0.0;
    }
}

__global__ void k_extract_cellID(particle_t *m_particles, int n, int *cellID_keys)
{
    IRIS_CUDA_SETUP_WS(n);
    cellID_keys[from] = m_particles[from].cellID;
}

void fmm::load_particles_gpu()
{
    // Find the total amount of local charges. This is just a simple sum of ncharges from all incoming rank.
    m_nparticles = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	m_nparticles += m_iris->m_ncharges[rank];
    }

    // Allocate GPU memory for m_particles (and m_xparticles, it will be at the end of m_particles)
    // Also, allocate same sized array for the atom types (=1 if halo atom, =0 if own atom)
    m_particles = (particle_t *)memory::wmalloc_gpu_cap((void *)m_particles, m_nparticles, sizeof(particle_t), &m_npart_cap);
    m_keys = (int *)memory::wmalloc_gpu_cap((void *)m_keys, m_nparticles, sizeof(int), &m_keys_cap);

    // Allocate GPU memory for the charges coming from all client ranks
    // This is done all at once so as no to interfere with mem transfer/kernel overlapping in the next loop
    for(int rank = 0; rank < m_iris->m_client_size; rank++) {
	int ncharges = m_iris->m_ncharges[rank];
	iris_real *charges = m_iris->m_charges[rank];
	m_charges_gpu[rank] = (iris_real *)memory::wmalloc_gpu_cap((void *)m_charges_gpu[rank], ncharges, 5*sizeof(iris_real), &m_charges_gpu_cap[rank]);
    }

    // Start the particle loading itself. The <H2D, kernel> pairs runs on separate streams and thus overlaps memory transfer for more than 1 source rank
    int offset = cell_meta_t::offset_for_level(max_level());
    int nd = 1 << max_level();
    int hwm = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	int ncharges = m_iris->m_ncharges[rank];
	iris_real *charges = m_iris->m_charges[rank];
	cudaMemcpyAsync(m_charges_gpu[rank], charges, ncharges * 5 * sizeof(iris_real), cudaMemcpyDefault, m_streams[rank % IRIS_CUDA_FMM_NUM_STREAMS]);
	
	int nthreads = MIN(IRIS_CUDA_NTHREADS, ncharges);
	int nblocks = IRIS_CUDA_NBLOCKS(ncharges, nthreads);
	k_load_charges<<<nblocks, nthreads, 0, m_streams[rank % IRIS_CUDA_FMM_NUM_STREAMS]>>>(m_charges_gpu[rank], ncharges, hwm,
											      m_domain->m_global_box.xlo, m_domain->m_global_box.ylo, m_domain->m_global_box.zlo,
											      m_leaf_size[0], m_leaf_size[1], m_leaf_size[2],
											      max_level(), offset, m_particles, rank);
	hwm += ncharges;
    }
        
    cudaDeviceSynchronize();  // all k_load_charges kernels must have finished to have valid m_particles

    // Get the cellIDs in the reordered array
    int nthreads = MIN(IRIS_CUDA_NTHREADS, m_nparticles);
    int nblocks = IRIS_CUDA_NBLOCKS(m_nparticles, nthreads);
    k_extract_cellID<<<nblocks, nthreads, 0, m_streams[0]>>>(m_particles, m_nparticles, m_keys);
    cudaStreamSynchronize(m_streams[0]);  // we want the above to finish before we do the sorts below
    
    // We now need to sort the m_particles by cellID
    thrust::device_ptr<int>         keys(m_keys);
    thrust::device_ptr<particle_t>  part(m_particles);
    thrust::sort_by_key(thrust::cuda::par.on(m_streams[0]), keys, keys+m_nparticles, part);
    
    m_logger->info("This rank owns %d particles", m_nparticles);
}


/////////////////////////////
// Distribute particles v1 //
/////////////////////////////


__device__ void d_compute_com(particle_t *in_particles, int num_points, int first_child, cell_t *out_target)
{
    // iris_real M = 0.0;
    // for(int i=0;i<num_points;i++) {
    // 	M += in_particles[first_child+i].xyzq[3];
    // }
    for(int i=0;i<num_points;i++) {
    	out_target->ses.c.r[0] += in_particles[first_child+i].xyzq[0]; // * in_particles[first_child+i].xyzq[3];
    	out_target->ses.c.r[1] += in_particles[first_child+i].xyzq[1]; // * in_particles[first_child+i].xyzq[3];
    	out_target->ses.c.r[2] += in_particles[first_child+i].xyzq[2]; // * in_particles[first_child+i].xyzq[3];
    }

    out_target->ses.c.r[0] /= num_points;
    out_target->ses.c.r[1] /= num_points;
    out_target->ses.c.r[2] /= num_points;

    iris_real max_dist2 = 0.0;
    for(int i=0;i<num_points;i++) {
	iris_real dx = in_particles[first_child+i].xyzq[0] - out_target->ses.c.r[0];
	iris_real dy = in_particles[first_child+i].xyzq[1] - out_target->ses.c.r[1];
	iris_real dz = in_particles[first_child+i].xyzq[2] - out_target->ses.c.r[2];
	iris_real dist2 = dx*dx + dy*dy + dz*dz;
	if(dist2 > max_dist2) {
	    max_dist2 = dist2;
	}
    }
    out_target->ses.r = sqrt(max_dist2);
}

__device__ int d_bsearch(particle_t *in_particles, int in_count, int cellID)
{
    int start = 0;
    int end = in_count-1;
    while (start <= end) {
        int m = start + (end - start) / 2;
        if (in_particles[m].cellID == cellID) {
            return m;
	}else if (in_particles[m].cellID < cellID) {
            start = m + 1;
	}else {
	    end = m - 1;
	}
    }
    return -1; 
}

__global__ void k_distribute_particles(particle_t *in_particles, int in_count, int in_flags, cell_t *out_target, int offset, int *m_max_particles_gpu)
{
    int tid = IRIS_CUDA_TID;
    int cellID = offset + tid;

    int from = d_bsearch(in_particles, in_count, cellID);
    if(from == -1) {
    	return;
    }
    int to = from;
    
    while(from > 0 && in_particles[from].cellID >= cellID)       { from--; }
    while(from < in_count && in_particles[from].cellID < cellID) { from++; }
    while(to < in_count-1 && in_particles[to].cellID <= cellID)  { to++; }
    while(to >= 0 && in_particles[to].cellID > cellID)           { to--; }
    
    int num_children = (to - from + 1);
    if(num_children <= 0) {
	return;
    }
    out_target[cellID].first_child = from;
    out_target[cellID].num_children = num_children;
    out_target[cellID].flags = in_flags;
    d_compute_com(in_particles, num_children, from, out_target+cellID);
    atomicMax(m_max_particles_gpu, num_children);
}


void fmm::distribute_particles_gpu_v1(particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    if(in_count == 0) {
	return;
    }
    
    int nleafs = (1 << 3 * max_level());
    int offset = cell_meta_t::offset_for_level(max_level());
    int nthreads = MIN(IRIS_CUDA_NTHREADS, nleafs);
    int nblocks = IRIS_CUDA_NBLOCKS(nleafs, nthreads);
    k_distribute_particles<<<nblocks, nthreads, 0, m_streams[0]>>>(in_particles, in_count, in_flags, out_target, offset, m_max_particles_gpu);
    cudaMemcpyAsync(&m_max_particles, m_max_particles_gpu, sizeof(int), cudaMemcpyDefault, m_streams[0]); // TODO: make this async
}


//////////////////////////
// Distribute particles //
//////////////////////////


__global__ void k_init_first_child(cell_t *out_target, int offset)
{
    out_target[IRIS_CUDA_TID + offset].first_child = INT_MAX;
}


__global__ void k_find_range(particle_t *in_particles, int in_count, cell_t *out_target, int tile_size, int tile_offset, int in_flags)
{
    int i = IRIS_CUDA_TID;
    if(i >= tile_size) {
	return;
    }
    
    i += tile_offset;  // particle index
    if(i >= in_count) {
	return;
    }

    int cellID = in_particles[i].cellID;
    atomicMin(&(out_target[cellID].first_child), i);
    atomicAdd(&(out_target[cellID].num_children), 1);
    out_target[cellID].flags = in_flags;
 
    // prepare to find the center of mass
    atomicAdd(&(out_target[cellID].ses.c.r[0]), in_particles[i].xyzq[0]);
    atomicAdd(&(out_target[cellID].ses.c.r[1]), in_particles[i].xyzq[1]);
    atomicAdd(&(out_target[cellID].ses.c.r[2]), in_particles[i].xyzq[2]);
}

__global__ void k_find_max_particles(cell_t *out_target, int offset, int *m_max_particles_gpu)
{
    int tid = IRIS_CUDA_TID;
    int cellID = offset + tid;
    cell_t *leaf = &(out_target[cellID]);
    atomicMax(m_max_particles_gpu, leaf->num_children);
}


__global__ void k_find_ses(particle_t *in_particles, int in_flags, cell_t *out_target, int offset)
{
    int tid = IRIS_CUDA_TID;
    int cellID = offset + tid;
    cell_t *leaf = &(out_target[cellID]);
    int num_points = leaf->num_children;

    leaf->ses.c.r[0] /= num_points;
    leaf->ses.c.r[1] /= num_points;
    leaf->ses.c.r[2] /= num_points;

    iris_real max_dist2 = 0.0;
    for(int i=0;i<num_points;i++) {
    	iris_real dx = in_particles[leaf->first_child+i].xyzq[0] - leaf->ses.c.r[0];
    	iris_real dy = in_particles[leaf->first_child+i].xyzq[1] - leaf->ses.c.r[1];
    	iris_real dz = in_particles[leaf->first_child+i].xyzq[2] - leaf->ses.c.r[2];
    	iris_real dist2 = dx*dx + dy*dy + dz*dz;
    	if(dist2 > max_dist2) {
    	    max_dist2 = dist2;
    	}
    }
    leaf->ses.r = sqrt(max_dist2);
}

void fmm::distribute_particles_gpu(particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    if(in_count == 0) {
	return;
    }

    // First, set first_child of all leafs to INT_MAX to facilitate MIN in the next kernel
    int nleafs = (1 << 3 * max_level());
    int offset = cell_meta_t::offset_for_level(max_level());
    int nthreads = MIN(IRIS_CUDA_NTHREADS, nleafs);
    int nblocks = IRIS_CUDA_NBLOCKS(nleafs, nthreads);
    k_init_first_child<<<nblocks, nthreads>>>(out_target, offset);

    // Then, find the first_child and num_children for each leaf
    // Also, sum all particle coordinates for each cell to prepare to find the center of mass
    // Do this in several streams to reduce atomic conflicts inside threads
    int nstreams = 4;
    int tile_offset;
    int tile_size = in_count / nstreams + ((in_count % nstreams)?1:0);
    dim3 nthreads2(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks2((tile_size-1)/IRIS_CUDA_NTHREADS + 1, 1, 1);

    for(int i=0;i<nstreams;i++) {
	tile_offset = i * tile_size;
	k_find_range<<<nblocks2, nthreads2, 0, m_streams[i]>>>(in_particles, in_count, out_target, tile_size, tile_offset, in_flags);
    }
    cudaDeviceSynchronize();
    
    k_find_ses<<<nblocks, nthreads, 0, m_streams[0]>>>(in_particles, in_flags, out_target, offset);
    k_find_max_particles<<<nblocks, nthreads, 0, m_streams[1]>>>(out_target, offset, m_max_particles_gpu);
    cudaMemcpy(&m_max_particles, m_max_particles_gpu, sizeof(int), cudaMemcpyDefault);
}


//////////////////////////////
// Distribute xparticles v1 //
//////////////////////////////


__device__ int d_xbsearch(xparticle_t *in_particles, int in_count, int cellID)
{
    int start = 0;
    int end = in_count-1;
    while (start <= end) {
        int m = start + (end - start) / 2;
        if (in_particles[m].cellID == cellID) {
            return m;
	}else if (in_particles[m].cellID < cellID) {
            start = m + 1;
	}else {
	    end = m - 1;
	}
    }
    return -1; 
}

__global__ void k_distribute_xparticles(xparticle_t *in_particles, int in_count, int in_flags, cell_t *out_target, int offset, int *m_max_particles_gpu)
{
    int tid = IRIS_CUDA_TID;
    int cellID = offset + tid;
    int from = d_xbsearch(in_particles, in_count, cellID);
    if(from == -1) {
    	return;
    }
    int to = from;
    
    while(from > 0 && in_particles[from].cellID >= cellID)       { from--; }
    while(from < in_count && in_particles[from].cellID < cellID) { from++; }
    while(to < in_count-1 && in_particles[to].cellID <= cellID)  { to++; }
    while(to >= 0 && in_particles[to].cellID > cellID)           { to--; }
    
    int num_children = (to - from + 1);
    if(num_children <= 0) {
	return;
    }
    out_target[cellID].first_child = from;
    out_target[cellID].num_children = num_children;
    out_target[cellID].flags = in_flags;
    atomicMax(m_max_particles_gpu, num_children);
}

void fmm::distribute_xparticles_gpu_v1(xparticle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    if(in_count == 0) {
	return;
    }
    
    int nleafs = (1 << 3 * max_level());
    int offset = cell_meta_t::offset_for_level(max_level());
    int nthreads = MIN(IRIS_CUDA_NTHREADS, nleafs);
    int nblocks = IRIS_CUDA_NBLOCKS(nleafs, nthreads);
    k_distribute_xparticles<<<nblocks, nthreads, 0, m_streams[0]>>>(in_particles, in_count, in_flags, out_target, offset, m_max_particles_gpu);
    cudaMemcpyAsync(&m_max_particles, m_max_particles_gpu, sizeof(int), cudaMemcpyDefault, m_streams[0]); // TODO: make this async
}


///////////////////////////
// Distribute xparticles //
///////////////////////////


__global__ void k_find_xrange(xparticle_t *in_particles, int in_count, cell_t *out_target, int tile_size, int tile_offset, int in_flags)
{
    int i = IRIS_CUDA_TID;
    if(i >= tile_size) {
	return;
    }
    
    i += tile_offset;  // particle index
    if(i >= in_count) {
	return;
    }

    int cellID = in_particles[i].cellID;
    atomicMin(&(out_target[cellID].first_child), i);
    atomicAdd(&(out_target[cellID].num_children), 1);
    out_target[cellID].flags = in_flags;
}


void fmm::distribute_xparticles_gpu(xparticle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    if(in_count == 0) {
	return;
    }

    cudaDeviceSynchronize();
    
    // Then, find the first_child and num_children for each leaf
    // Also, sum all particle coordinates for each cell to prepare to find the center of mass
    // Do this in several streams to reduce atomic conflicts inside threads
    int nstreams = 4;
    int tile_offset;
    int tile_size = in_count / nstreams + ((in_count % nstreams)?1:0);
    dim3 nthreads2(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks2((tile_size-1)/IRIS_CUDA_NTHREADS + 1, 1, 1);

    for(int i=0;i<nstreams;i++) {
	tile_offset = i * tile_size;
	k_find_xrange<<<nblocks2, nthreads2, 0, m_streams[i]>>>(in_particles, in_count, out_target, tile_size, tile_offset, in_flags);
    }
    cudaDeviceSynchronize();

    int nleafs = (1 << 3 * max_level());
    int offset = cell_meta_t::offset_for_level(max_level());
    int nthreads = MIN(IRIS_CUDA_NTHREADS, nleafs);
    int nblocks = IRIS_CUDA_NBLOCKS(nleafs, nthreads);
    k_find_max_particles<<<nblocks, nthreads, 0, m_streams[1]>>>(out_target, offset, m_max_particles_gpu);
    cudaMemcpy(&m_max_particles, m_max_particles_gpu, sizeof(int), cudaMemcpyDefault);
}


//////////////////
// Link parents //
//////////////////


__global__ void k_link_parents_proper(cell_t *io_cells, int start, int end)
{
    IRIS_CUDA_SETUP_WS(end-start);
    
    for(int j=start+from;j<start+to;j++) {
	if((io_cells[j].num_children != 0) ||                   // cell is a non-empty leaf
	   (io_cells[j].flags & IRIS_FMM_CELL_HAS_CHILDREN) ||  // or cell is a non-leaf and has some children
	   (io_cells[j].flags & IRIS_FMM_CELL_ALIEN_NL)) {        // or is an alien cell
	    int parent = cell_meta_t::parent_of(j);
	    atomicOr(&io_cells[parent].flags, IRIS_FMM_CELL_HAS_CHILD1 << ((j - start) % 8));
	}
    }
}

__global__ void k_compute_ses_nl(cell_t *io_cells, int start, int end)
{
    IRIS_CUDA_SETUP_WS(end-start);
    for(int j=start+from;j<start+to;j++) {
	if(io_cells[j].ses.r != 0.0) {
	    continue;
	}
	sphere_t S[8];
	int ns = 0;
	for(int k = 0;k<8;k++) {
	    int mask = IRIS_FMM_CELL_HAS_CHILD1 << k;
	    if(io_cells[j].flags & mask) {
		int childID = end + 8*(j-start) + k;
		S[ns].c.r[0] = io_cells[childID].ses.c.r[0];
		S[ns].c.r[1] = io_cells[childID].ses.c.r[1];
		S[ns].c.r[2] = io_cells[childID].ses.c.r[2];
		S[ns].r = io_cells[childID].ses.r;
		ns++;
	    }
	}
	ses_of_spheres(S, ns, &(io_cells[j].ses));
    }
}
    
void fmm::link_parents_gpu(cell_t *io_cells)
{
    for(int i=max_level();i>0;i--) {
	int start = cell_meta_t::offset_for_level(i);
	int end = cell_meta_t::offset_for_level(i+1);
	int n = end - start;
	int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
	int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
	k_link_parents_proper<<<nblocks, nthreads, 0, m_streams[0]>>>(io_cells, start, end);
    }

    for(int i=max_level()-1;i>=0;i--) {
    	int start = cell_meta_t::offset_for_level(i);
    	int end = cell_meta_t::offset_for_level(i+1);
    	int n = end - start;
    	int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    	int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
	k_compute_ses_nl<<<nblocks, nthreads, 0, m_streams[0]>>>(io_cells, start, end);
    }
}


//////////////
// Eval M2M //
//////////////


__global__ void k_eval_m2m(cell_t *in_cells, bool invalid_only, int offset, int children_offset, iris_real *m_M, int m_nterms, int m_order)
{
    iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+1)];
    
    int tid = IRIS_CUDA_TID;
    int tcellID = tid + offset;
    int j = blockIdx.y;
    
    if(invalid_only && (in_cells[tcellID].flags & IRIS_FMM_CELL_VALID_M)) {
	return;
    }

    if(!(in_cells[tcellID].flags & (IRIS_FMM_CELL_HAS_CHILD1 << j))) {
	return;
    }
    
    iris_real cx = in_cells[tcellID].ses.c.r[0];
    iris_real cy = in_cells[tcellID].ses.c.r[1];
    iris_real cz = in_cells[tcellID].ses.c.r[2];
    
    iris_real *M = m_M + tcellID * m_nterms;
    
    int scellID = children_offset + 8*tid + j;
    iris_real x = in_cells[scellID].ses.c.r[0] - cx;
    iris_real y = in_cells[scellID].ses.c.r[1] - cy;
    iris_real z = in_cells[scellID].ses.c.r[2] - cz;
    
    m2m(m_order, x, y, z, m_M + scellID * m_nterms, M, scratch);
    in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_M;
}

void fmm::eval_m2m_gpu(cell_t *in_cells, bool invalid_only)
{
    cudaStreamSynchronize(m_streams[0]);  // wait for link parents
    cudaStreamSynchronize(m_streams[1]);  // wait for p2m
    int from, to;
    if(invalid_only) {
	from = m_local_root_level-1;
	to = 0;
    }else {
	from = max_level()-1;
	to = m_local_root_level;
    }
    
    for(int level = from;level>=to;level--) {
	int start = cell_meta_t::offset_for_level(level);
	int end = cell_meta_t::offset_for_level(level+1);
	int n = end - start;
	dim3 nthreads(MIN(IRIS_CUDA_NTHREADS, n), 1, 1);
	dim3 nblocks((n-1)/IRIS_CUDA_NTHREADS+1, 8, 1);
	k_eval_m2m<<<nblocks, nthreads, 0, m_streams[1]>>>(in_cells, invalid_only, start, end, m_M, m_nterms, m_order);
    }
}


////////////////////
// Relink parents //
////////////////////


__global__ void k_clear_nl_children(cell_t *io_cells, int count)
{
    int tid = IRIS_CUDA_TID;
    if(tid < count) {
	io_cells[tid].flags &= ~IRIS_FMM_CELL_HAS_CHILDREN;
    }
}

__global__ void k_clear_nl_ses(cell_t *io_cells, int count)
{
    int tid = IRIS_CUDA_TID;
    if(tid < count) {
	io_cells[tid].ses.r = 0.0;
    }
}


void fmm::relink_parents_gpu(cell_t *io_cells)
{
    int end = cell_meta_t::offset_for_level(max_level());
    int nthreads = MIN(IRIS_CUDA_NTHREADS, end);
    int nblocks = IRIS_CUDA_NBLOCKS(end, nthreads);
    k_clear_nl_children<<<nblocks, nthreads>>>(io_cells, end);

    end = cell_meta_t::offset_for_level(m_local_root_level);
    nthreads = MIN(IRIS_CUDA_NTHREADS, end);
    nblocks = IRIS_CUDA_NBLOCKS(end, nthreads);
    k_clear_nl_ses<<<nblocks, nthreads>>>(io_cells, end);

    link_parents_gpu(io_cells);
}


//////////////
// Eval M2L //
//////////////


__global__ void k_eval_m2l(interact_item_t *list, int list_size, cell_t *m_cells, cell_t *m_xcells, 
			   iris_real gxsize, iris_real gysize, iris_real gzsize, int m_nterms, int m_order, iris_real *m_M, iris_real *m_L)
{
    iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+1)];
    
    int tid = IRIS_CUDA_TID;
    if(tid >= list_size) {
	return;
    }

    int srcID = list[tid].sourceID;
    int destID = list[tid].targetID;
    iris_real xoff = list[tid].ix * gxsize;
    iris_real yoff = list[tid].iy * gysize;
    iris_real zoff = list[tid].iz * gzsize;
    
    iris_real sx = m_xcells[srcID].ses.c.r[0] + xoff;
    iris_real sy = m_xcells[srcID].ses.c.r[1] + yoff;
    iris_real sz = m_xcells[srcID].ses.c.r[2] + zoff;

    iris_real tx = m_cells[destID].ses.c.r[0];
    iris_real ty = m_cells[destID].ses.c.r[1];
    iris_real tz = m_cells[destID].ses.c.r[2];

    iris_real x = tx - sx;
    iris_real y = ty - sy;
    iris_real z = tz - sz;

    bool do_other_side = (list[tid].ix == 0 && list[tid].iy == 0 && list[tid].iz == 0);

    do_other_side = false;
    
    memset(scratch, 0, m_nterms*sizeof(iris_real));
    m2l_v2(m_order, x, y, z, m_M + srcID * m_nterms, m_L + destID * m_nterms, scratch,
	   m_M + destID * m_nterms, m_L + srcID * m_nterms, do_other_side);

    atomicOr(&(m_cells[destID].flags), IRIS_FMM_CELL_VALID_L);
    if(do_other_side) {
	atomicOr(&(m_cells[srcID].flags), IRIS_FMM_CELL_VALID_L);
    }
}

void fmm::eval_m2l_gpu()
{
    int n = m_m2l_list.size();
    if(n == 0) {
	return;
    }

    m_m2l_list_gpu = (interact_item_t *)memory::wmalloc_gpu_cap(m_m2l_list_gpu, n, sizeof(interact_item_t), &m_m2l_list_cap);
    cudaMemcpyAsync(m_m2l_list_gpu, m_m2l_list.data(), n * sizeof(interact_item_t), cudaMemcpyDefault, m_streams[0]);
    cudaEventRecord(m_m2l_memcpy_done, m_streams[0]);
    
    int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_eval_m2l<<<nblocks, nthreads, 0, m_streams[0]>>>(m_m2l_list_gpu, n, m_cells, m_xcells, 
    						       m_domain->m_global_box.xsize, m_domain->m_global_box.ysize, m_domain->m_global_box.zsize, m_nterms,
    						       m_order, m_M, m_L);
    cudaEventSynchronize(m_m2l_memcpy_done);
    m_m2l_list.clear();
}


//////////////
// Eval L2L //
//////////////


__global__ void k_eval_l2l(cell_t *m_cells, int offset, int children_offset, iris_real *m_L, int m_nterms, int m_order)
{
    iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+1)];
    
    int tid = IRIS_CUDA_TID;
    int scellID = tid + offset;
    int j = blockIdx.y;

    if(scellID >= children_offset) {
	return;
    }
    
    if(!(m_cells[scellID].flags & IRIS_FMM_CELL_VALID_L)) {
	return;
    }

    if(!(m_cells[scellID].flags & (IRIS_FMM_CELL_HAS_CHILD1 << j))) {
	return;
    }
    
    iris_real cx = m_cells[scellID].ses.c.r[0];
    iris_real cy = m_cells[scellID].ses.c.r[1];
    iris_real cz = m_cells[scellID].ses.c.r[2];

    iris_real *L = m_L + scellID * m_nterms;
    
    int tcellID = children_offset + 8*tid + j;
    iris_real x = cx - m_cells[tcellID].ses.c.r[0];
    iris_real y = cy - m_cells[tcellID].ses.c.r[1];
    iris_real z = cz - m_cells[tcellID].ses.c.r[2];
    
    l2l(m_order, x, y, z, L, m_L + tcellID * m_nterms, scratch);
    m_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_L;
}

void fmm::eval_l2l_gpu()
{
    for(int level = 0; level < m_depth-1; level++) {
	int start = cell_meta_t::offset_for_level(level);
	int end = cell_meta_t::offset_for_level(level+1);
	int n = end - start;
	dim3 nthreads(MIN(IRIS_CUDA_NTHREADS, n), 1, 1);
	dim3 nblocks((n-1)/IRIS_CUDA_NTHREADS+1, 8, 1);
	k_eval_l2l<<<nblocks, nthreads, 0, m_streams[0]>>>(m_cells, start, end, m_L, m_nterms, m_order);
    }
}


//////////////
// Eval L2P //
//////////////

__global__ void k_eval_l2p(cell_t *m_cells, int offset, particle_t *m_particles, int m_order, iris_real *m_L, int m_nterms)
{
    iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+1)];
    iris_real scratch_size = m_nterms * sizeof(iris_real);
    
    int leaf_idx = blockIdx.y * gridDim.z + blockIdx.z;   // Which cell we are processing
    int cellID = leaf_idx + offset;
    int j = IRIS_CUDA_TID;                                // Target particle inside cellID

    cell_t *leaf = m_cells + cellID;
    int npart = leaf->num_children;                       // Number of particles in the cell
     
    if(j >= npart || !(leaf->flags & IRIS_FMM_CELL_VALID_L)) {
	return;
    }

    particle_t *part = m_particles + leaf->first_child + j;

    iris_real *L = m_L + cellID * m_nterms;
    
    iris_real x = leaf->ses.c.r[0] - part->xyzq[0];
    iris_real y = leaf->ses.c.r[1] - part->xyzq[1];
    iris_real z = leaf->ses.c.r[2] - part->xyzq[2];
    iris_real q = part->xyzq[3];
    
    iris_real phi, Ex, Ey, Ez;
    
    l2p(m_order, x, y, z, q, L, scratch, &phi, &Ex, &Ey, &Ez);

    part->tgt[0] += phi;
    part->tgt[1] += Ex;
    part->tgt[2] += Ey;
    part->tgt[3] += Ez;
}

void fmm::eval_l2p_gpu()
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;
    
    dim3 nthreads(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks((m_max_particles-1)/IRIS_CUDA_NTHREADS + 1, nleafs, 1);
    k_eval_l2p<<<nblocks, nthreads, 0, m_streams[0]>>>(m_cells, offset, m_particles, m_order, m_L, m_nterms);
}


///////////////////////////////
// Compute energy and virial //
///////////////////////////////


// TODO: compute virial
__global__ void k_compute_energy_and_virial(particle_t *m_particles, iris_real *out_ener, int npart)
{
    __shared__ iris_real ener_acc[IRIS_CUDA_NTHREADS];
    __shared__ iris_real vir_acc[IRIS_CUDA_NTHREADS][6];
    int iacc = threadIdx.x;
    ener_acc[iacc] = 0.0;
    
    int tid = IRIS_CUDA_TID;
    if(tid == 0) {
	*out_ener = 0.0;
    }
    if(tid < npart) {
    ener_acc[iacc] += m_particles[tid].tgt[0] * m_particles[tid].xyzq[3];
    iris_real xfx = m_particles[tid].xyzq[0] * m_particles[tid].tgt[1];
    iris_real yfx = m_particles[tid].xyzq[1] * m_particles[tid].tgt[1];
    iris_real zfx = m_particles[tid].xyzq[2] * m_particles[tid].tgt[1];
    iris_real xfy = m_particles[tid].xyzq[0] * m_particles[tid].tgt[2];
    iris_real yfy = m_particles[tid].xyzq[1] * m_particles[tid].tgt[2];
    iris_real zfy = m_particles[tid].xyzq[2] * m_particles[tid].tgt[2];
    iris_real xfz = m_particles[tid].xyzq[0] * m_particles[tid].tgt[3];
    iris_real yfz = m_particles[tid].xyzq[1] * m_particles[tid].tgt[3];
    iris_real zfz = m_particles[tid].xyzq[2] * m_particles[tid].tgt[3];
    vir_acc[iacc][0] += xfx;
    vir_acc[iacc][1] += yfy;
    vir_acc[iacc][2] += zfz;
    vir_acc[iacc][3] += (xfy + yfx);
    vir_acc[iacc][4] += (xfz + zfx);
    vir_acc[iacc][5] += (yfz + zfy);
    }

    __syncthreads();

    for(int i=blockDim.x; i>0; i/=2) {
	int stride = blockDim.x/i;
	if(iacc < (blockDim.x - stride) && iacc % (2*stride) == 0) {
        ener_acc[iacc] += ener_acc[iacc+stride];
        vir_acc[iacc][0] += vir_acc[iacc+stride][0];
        vir_acc[iacc][1] += vir_acc[iacc+stride][1];
        vir_acc[iacc][2] += vir_acc[iacc+stride][2];
        vir_acc[iacc][3] += vir_acc[iacc+stride][3];
        vir_acc[iacc][4] += vir_acc[iacc+stride][4];
        vir_acc[iacc][5] += vir_acc[iacc+stride][5];
	}
	__syncthreads();
    }
    if(iacc == 0) {
    atomicAdd(out_ener, ener_acc[0]);
    atomicAdd(out_ener+1, vir_acc[0][0]);
    atomicAdd(out_ener+2, vir_acc[0][1]);
    atomicAdd(out_ener+3, vir_acc[0][2]);
    atomicAdd(out_ener+4, vir_acc[0][3]);
    atomicAdd(out_ener+5, vir_acc[0][4]);
    atomicAdd(out_ener+6, vir_acc[0][5]);
    }
}

void fmm::compute_energy_and_virial_gpu()
{
    cudaDeviceSynchronize();
    
    int n = m_nparticles;
    int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_compute_energy_and_virial<<<nblocks, nthreads, 0, m_streams[0]>>>(m_particles, m_evir_gpu, n);
    cudaMemcpyAsync(&(m_iris->m_Ek), m_evir_gpu, 7*sizeof(iris_real), cudaMemcpyDefault, m_streams[0]);
    cudaStreamSynchronize(m_streams[0]); // must be synchronous
    m_iris->m_Ek *= 0.5 * m_units->ecf;
    m_iris->m_virial[0] *= 0.5 * m_units->ecf;
	m_iris->m_virial[1] *= 0.5 * m_units->ecf;
	m_iris->m_virial[2] *= 0.5 * m_units->ecf;
	m_iris->m_virial[3] *= 0.25* m_units->ecf; //make the virial symmetric - multipling by extra 0.5 commumig from the averaging offdiagonal elementes 
	m_iris->m_virial[4] *= 0.25* m_units->ecf;
	m_iris->m_virial[5] *= 0.25* m_units->ecf;
}


//////////////////////
// Send back forces //
//////////////////////


__global__ void k_extract_rank(particle_t *m_particles, int n, int *keys)
{
    IRIS_CUDA_SETUP_WS(n);
    keys[from] = m_particles[from].rank;
}

void fmm::send_back_forces_gpu()
{
    thrust::device_ptr<int>         keys(m_keys);
    thrust::device_ptr<particle_t>  part(m_particles);

    int nthreads = MIN(IRIS_CUDA_NTHREADS, m_nparticles);
    int nblocks = IRIS_CUDA_NBLOCKS(m_nparticles, nthreads);
    k_extract_rank<<<nblocks, nthreads, 0, m_streams[0]>>>(m_particles, m_nparticles, m_keys);
    thrust::sort_by_key(thrust::cuda::par.on(m_streams[0]), keys, keys+m_nparticles, part);

    m_particles_cpu = (particle_t *)memory::wmalloc_gpu_cap(m_particles_cpu, m_nparticles, sizeof(particle_t), &m_particles_cpu_cap, true);
    cudaMemcpyAsync(m_particles_cpu, m_particles, m_nparticles * sizeof(particle_t), cudaMemcpyDefault, m_streams[0]);
    cudaStreamSynchronize(m_streams[0]); // must be sync
    send_back_forces_cpu(m_particles_cpu, false);
}

void fmm::cuda_specific_construct()
{
    for(int i=0;i<IRIS_CUDA_FMM_NUM_STREAMS;i++) {
	cudaStreamCreate(&m_streams[i]);
    }
    cudaEventCreate(&m_m2l_memcpy_done);
    cudaEventCreate(&m_p2p_memcpy_done);
    //cudaDeviceSetLimit(cudaLimitStackSize, 8192);  // otherwise distribute_particles won't work because of the welzl recursion
    cudaMalloc((void **)&m_evir_gpu, 7*sizeof(iris_real));
    cudaMalloc((void **)&m_max_particles_gpu, sizeof(int));

    // TODO: these must be deleted in a destruct (but not in fmm.cpp, which doesn't know about thrust potentially)
    m_a2a_cell_cnt_gpu = new thrust::device_vector<int>();
    m_a2a_cell_disp_gpu = new thrust::device_vector<int>();
    m_a2a_sendbuf_gpu = new thrust::device_vector<xparticle_t>();

    IRIS_CUDA_CHECK_ERROR;
}

#endif
