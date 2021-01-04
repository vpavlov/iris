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


////////////////////
// Load Particles //
////////////////////


__global__ void k_load_charges(iris_real *charges, int ncharges, int hwm,
			       iris_real xlo, iris_real ylo, iris_real zlo,
			       iris_real lsx, iris_real lsy, iris_real lsz,
			       int max_level, int offset, particle_t *m_particles, int rank,
			       int *atom_types)
{
    IRIS_CUDA_SETUP_WS(ncharges);
    for(int i=from;i<to;i++) {
	iris_real tx = (charges[i * 5 + 0] - xlo) / lsx;
	iris_real ty = (charges[i * 5 + 1] - ylo) / lsy;
	iris_real tz = (charges[i * 5 + 2] - zlo) / lsz;

	int lc[3];
	lc[0] = (int) tx;
	lc[1] = (int) ty;
	lc[2] = (int) tz;
	
	int id = 0;
	for(int l=0;l<max_level; l++) {
	    for(int d=0;d<3;d++) {
		id += (lc[d] & 1) << (3*l + d);
		lc[d] >>= 1;
	    }
	}
	
	int cellID = offset + id;
	int chargeID = (int)charges[i*5 + 4];
	
	m_particles[i+hwm].rank = rank;
	m_particles[i+hwm].index = chargeID; // (chargeID > 0)?chargeID:-chargeID;
	m_particles[i+hwm].cellID = cellID;
	m_particles[i+hwm].xyzq[0] = charges[i*5+0];
	m_particles[i+hwm].xyzq[1] = charges[i*5+1];
	m_particles[i+hwm].xyzq[2] = charges[i*5+2];
	m_particles[i+hwm].xyzq[3] = charges[i*5+3];
	m_particles[i+hwm].tgt[0] = 0.0;
	m_particles[i+hwm].tgt[1] = 0.0;
	m_particles[i+hwm].tgt[2] = 0.0;
	m_particles[i+hwm].tgt[3] = 0.0;
	atom_types[i+hwm] = (chargeID > 0)?0:1;
    }
}

__global__ void k_extract_cellID(particle_t *m_particles, int n, int *cellID_keys)
{
    IRIS_CUDA_SETUP_WS(n);
    for(int i=from;i<to;i++) {
	cellID_keys[i] = m_particles[i].cellID;
    }
}

void fmm::load_particles_gpu()
{
    // Find the total amount of local charges, including halo atoms
    // This is just a simple sum of ncharges from all incoming rank
    int total_local_charges = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	total_local_charges += m_iris->m_ncharges[rank];
    }

    // Allocate GPU memory for m_particles (and m_xparticles, it will be at the end of m_particles)
    // Also, allocate same sized array for the atom types (=1 if halo atom, =0 if own atom)
    m_particles = (particle_t *)memory::wmalloc_gpu_cap((void *)m_particles, total_local_charges, sizeof(particle_t), &m_npart_cap);
    m_atom_types = (int *)memory::wmalloc_gpu_cap((void *)m_atom_types, total_local_charges, sizeof(int), &m_at_cap);
    m_cellID_keys = (int *)memory::wmalloc_gpu_cap((void *)m_cellID_keys, total_local_charges, sizeof(int), &m_cellID_keys_cap);

    // Allocate GPU memory for the charges coming from all client ranks
    // This is done all at once so as no to interfere with mem transfer/kernel overlapping in the next loop
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
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
											      max_level(), offset, m_particles, rank,
											      m_atom_types);
	cudaError_t err = cudaGetLastError();
	IRIS_CUDA_HANDLE_ERROR(err);
	hwm += ncharges;
    }
    
        
    cudaDeviceSynchronize();  // all k_load_charges kernels must have finished to have valid m_particles

    // At this point we have the m_particles filled up, with mixed halo/own atoms, etc.
    // We need to sort them according to the m_atom_type keys array and split into m_particles and m_xparticles
    thrust::device_ptr<int>         keys(m_atom_types);
    thrust::device_ptr<particle_t>  values(m_particles);
    thrust::sort_by_key(keys, keys+total_local_charges, values);

    // Now the first part of m_particles contains local atoms; second contains halo atoms
    // Number of local particles can be taken from iris: num_local_atoms
    // Number of halo particles is total_local_charges - num_local_atoms
    //m_nparticles = m_iris->num_local_atoms();
    m_nparticles = thrust::count(keys, keys+total_local_charges, 0);
    m_nxparticles = total_local_charges - m_nparticles;
    m_xparticles = m_particles + m_nparticles;

    // Get the cellIDs in the reordered array
    int nthreads = MIN(IRIS_CUDA_NTHREADS, total_local_charges);
    int nblocks = IRIS_CUDA_NBLOCKS(total_local_charges, nthreads);
    k_extract_cellID<<<nblocks, nthreads>>>(m_particles, total_local_charges, m_cellID_keys);

    // We now need to sort the m_particles and m_xparticles arrays by cellID
    thrust::device_ptr<int>         keys2(m_cellID_keys);
    thrust::device_ptr<particle_t>  part(m_particles);
    thrust::device_ptr<particle_t>  xpart(m_xparticles);
    thrust::sort_by_key(keys2, keys2+m_nparticles, part);
    thrust::sort_by_key(keys2+m_nparticles, keys2+total_local_charges, xpart);
    
    m_logger->info("FMM/GPU: This rank owns %d + %d halo particles", m_nparticles, m_nxparticles);
}


//////////////////////////
// Distribute particles //
//////////////////////////


// Smallest enclosing sphere by Welzl's algorithm -- but has a limited recursion depth...
__device__ void d_compute_ses(particle_t *in_particles, int num_points, int first_child, cell_t *out_target)
{
    point_t points[2*IRIS_MAX_NCRIT];
    for(int i=0;i<num_points;i++) {
    	points[i].r[0] = in_particles[first_child+i].xyzq[0];
    	points[i].r[1] = in_particles[first_child+i].xyzq[1];
    	points[i].r[2] = in_particles[first_child+i].xyzq[2];
    }
    ses_of_points(points, num_points, &(out_target->ses));
}

// Center of mass
__device__ void d_compute_com(particle_t *in_particles, int num_points, int first_child, cell_t *out_target)
{
    iris_real M = 0.0;
    for(int i=0;i<num_points;i++) {
	M += in_particles[first_child+i].xyzq[3];
    }
    for(int i=0;i<num_points;i++) {
	out_target->ses.c.r[0] += in_particles[first_child+i].xyzq[0] * in_particles[first_child+i].xyzq[3];
	out_target->ses.c.r[1] += in_particles[first_child+i].xyzq[1] * in_particles[first_child+i].xyzq[3];
	out_target->ses.c.r[2] += in_particles[first_child+i].xyzq[2] * in_particles[first_child+i].xyzq[3];
    }

    out_target->ses.c.r[0] /= M;
    out_target->ses.c.r[1] /= M;
    out_target->ses.c.r[2] /= M;

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

__global__ void k_distribute_particles(particle_t *in_particles, int in_count, int in_flags, cell_t *out_target, int offset, int nleafs)
{
    if(in_count == 0) {
	return;
    }
    
    int tid = IRIS_CUDA_TID;
    int cellID = offset + tid;
    float fract = (1.0*in_count)/nleafs;
    int from = (int)(fract * tid);
    int to = MIN((int)(fract * (tid + 1)), in_count-1);

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
    //d_compute_ses(in_particles, num_children, from, out_target+cellID);
    d_compute_com(in_particles, num_children, from, out_target+cellID);
}

void fmm::distribute_particles_gpu(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    int nleafs = (1 << 3 * max_level());
    int offset = cell_meta_t::offset_for_level(max_level());
    int nthreads = MIN(IRIS_CUDA_NTHREADS, nleafs);
    int nblocks = IRIS_CUDA_NBLOCKS(nleafs, nthreads);
    k_distribute_particles<<<nblocks, nthreads, 0, m_streams[0]>>>(in_particles, in_count, in_flags, out_target, offset, nleafs);
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
	    atomicAdd(&io_cells[parent].flags, IRIS_FMM_CELL_HAS_CHILD1 << ((j - start) % 8));
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
	k_link_parents_proper<<<nblocks, nthreads>>>(io_cells, start, end);
    }

    for(int i=max_level()-1;i>=0;i--) {
    	int start = cell_meta_t::offset_for_level(i);
    	int end = cell_meta_t::offset_for_level(i+1);
    	int n = end - start;
    	int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    	int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
	k_compute_ses_nl<<<nblocks, nthreads>>>(io_cells, start, end);
    }
}


//////////////
// Eval P2M //
//////////////


__global__ void k_eval_p2m(cell_t *in_cells, int offset, int end, bool alien_only, particle_t *m_particles, particle_t *m_xparticles, int m_order, iris_real *m_M, int m_nterms)
{
    int tid = IRIS_CUDA_TID;
    int cellID = tid + offset;
    if(cellID >= end) {
	return;
    }
    cell_t *leaf = &in_cells[cellID];
    
    // no particles here -- continue
    if(leaf->num_children == 0) {
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
    for(int j=0;j<leaf->num_children;j++) {
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
	p2m(m_order, x, y, z, q, M);
    }
    leaf->flags |= IRIS_FMM_CELL_VALID_M;
}

void fmm::eval_p2m_gpu(cell_t *in_cells, bool alien_only)
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int n = m_tree_size - offset;
    int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_eval_p2m<<<nblocks, nthreads>>>(in_cells, offset, m_tree_size, alien_only, m_particles, m_xparticles, m_order, m_M, m_nterms);
}


//////////////
// Eval M2M //
//////////////


__global__ void k_eval_m2m(cell_t *in_cells, bool invalid_only, int offset, int children_offset, iris_real *m_M, int m_nterms, int m_order, iris_real *scratch)
{
    int tid = IRIS_CUDA_TID;
    int tcellID = tid + offset;
    if(tid+offset >= children_offset) {
	return;
    }
    if(invalid_only && (in_cells[tcellID].flags & IRIS_FMM_CELL_VALID_M)) {
	return;
    }

    int scratch_size = 2*m_nterms*sizeof(iris_real);
    int scratch_offset = tid * 2 * m_nterms;

    iris_real cx = in_cells[tcellID].ses.c.r[0];
    iris_real cy = in_cells[tcellID].ses.c.r[1];
    iris_real cz = in_cells[tcellID].ses.c.r[2];
    
    bool valid_m = false;
    iris_real *M = m_M + tcellID * 2 * m_nterms;
    for(int j=0;j<8;j++) {
	int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
	if(!(in_cells[tcellID].flags & mask)) {
	    continue;
	}
	int scellID = children_offset + 8*tid + j;
	iris_real x = in_cells[scellID].ses.c.r[0] - cx;
	iris_real y = in_cells[scellID].ses.c.r[1] - cy;
	iris_real z = in_cells[scellID].ses.c.r[2] - cz;
	
	memset(scratch+scratch_offset, 0, scratch_size);
	m2m(m_order, x, y, z, m_M + scellID * 2 * m_nterms, M, scratch+scratch_offset);
	valid_m = true;
    }
    if(valid_m) {
	in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_M;
    }
}

void fmm::eval_m2m_gpu(cell_t *in_cells, bool invalid_only)
{
    int last_level = invalid_only ? 0 : m_local_root_level;
    for(int level = max_level()-1;level>=last_level;level--) {
	int start = cell_meta_t::offset_for_level(level);
	int end = cell_meta_t::offset_for_level(level+1);
	int n = end - start;
	int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
	int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
	k_eval_m2m<<<nblocks, nthreads>>>(in_cells, invalid_only, start, end, m_M, m_nterms, m_order, m_scratch);
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
    int nthreads = IRIS_CUDA_NTHREADS;
    int nblocks = IRIS_CUDA_NBLOCKS(end, nthreads);
    k_clear_nl_children<<<nblocks, nthreads>>>(io_cells, end);

    end = cell_meta_t::offset_for_level(m_local_root_level);
    nblocks = IRIS_CUDA_NBLOCKS(end, nthreads);
    k_clear_nl_ses<<<nblocks, nthreads>>>(io_cells, end);

    link_parents_gpu(io_cells);
}


//////////////
// Eval P2P //
//////////////


__global__ void k_eval_p2p(interact_item_t *list, int list_size, cell_t *m_cells, cell_t *m_xcells, particle_t *m_particles, particle_t *m_xparticles,
			   iris_real gxsize, iris_real gysize, iris_real gzsize)
{
    int tid = IRIS_CUDA_TID;
    if(tid >= list_size) {
	return;
    }
    int srcID = list[tid].sourceID;
    int destID = list[tid].targetID;
    iris_real xoff = list[tid].ix * gxsize;
    iris_real yoff = list[tid].iy * gysize;
    iris_real zoff = list[tid].iz * gzsize;
    for(int i=0;i<m_cells[destID].num_children;i++) {
	iris_real tx = m_particles[m_cells[destID].first_child + i].xyzq[0];
	iris_real ty = m_particles[m_cells[destID].first_child + i].xyzq[1];
	iris_real tz = m_particles[m_cells[destID].first_child + i].xyzq[2];

	iris_real sum_phi = 0.0;
	iris_real sum_ex = 0.0;
	iris_real sum_ey = 0.0;
	iris_real sum_ez = 0.0;
	for(int j=0;j<m_xcells[srcID].num_children;j++) {
	    iris_real sx, sy, sz, sq;
	    if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		sx = m_xparticles[m_xcells[srcID].first_child + j].xyzq[0] + xoff;
		sy = m_xparticles[m_xcells[srcID].first_child + j].xyzq[1] + yoff;
		sz = m_xparticles[m_xcells[srcID].first_child + j].xyzq[2] + zoff;
		sq = m_xparticles[m_xcells[srcID].first_child + j].xyzq[3];
	    }else {
		sx = m_particles[m_xcells[srcID].first_child + j].xyzq[0] + xoff;
		sy = m_particles[m_xcells[srcID].first_child + j].xyzq[1] + yoff;
		sz = m_particles[m_xcells[srcID].first_child + j].xyzq[2] + zoff;
		sq = m_particles[m_xcells[srcID].first_child + j].xyzq[3];
	    }

	    iris_real dx = tx - sx;
	    iris_real dy = ty - sy;
	    iris_real dz = tz - sz;
	    iris_real r2 = dx*dx + dy*dy + dz*dz;
	    iris_real inv_r2;
	    if(r2 == 0) {
		inv_r2 = 0;
	    }else {
		inv_r2 = 1/r2;
	    }
	    iris_real phi = sq * sqrt(inv_r2);
	    iris_real phi_over_r2 = phi * inv_r2;
	    iris_real ex = dx * phi_over_r2;
	    iris_real ey = dy * phi_over_r2;
	    iris_real ez = dz * phi_over_r2;

	    sum_phi += phi;
	    sum_ex += ex;
	    sum_ey += ey;
	    sum_ez += ez;
	}

	atomicAdd(m_particles[m_cells[destID].first_child + i].tgt + 0, sum_phi);
	atomicAdd(m_particles[m_cells[destID].first_child + i].tgt + 1, sum_ex);
	atomicAdd(m_particles[m_cells[destID].first_child + i].tgt + 2, sum_ey);
	atomicAdd(m_particles[m_cells[destID].first_child + i].tgt + 3, sum_ez);
    }
}

void fmm::eval_p2p_gpu()
{
    int n = m_p2p_list.size();
    m_logger->info("P2P size = %d", m_p2p_list.size());
    if(n == 0) {
	return;
    }

    m_p2p_list_gpu = (interact_item_t *)memory::wmalloc_gpu_cap(m_p2p_list_gpu, n, sizeof(interact_item_t), &m_p2p_list_cap);
    cudaMemcpyAsync(m_p2p_list_gpu, m_p2p_list.data(), n * sizeof(interact_item_t), cudaMemcpyDefault, m_streams[0]);
    
    int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_eval_p2p<<<nblocks, nthreads, 0, m_streams[0]>>>(m_p2p_list_gpu, n, m_cells, m_xcells, m_particles, m_xparticles,
    						       m_domain->m_global_box.xsize, m_domain->m_global_box.ysize, m_domain->m_global_box.zsize);
}


//////////////
// Eval M2L //
//////////////

__global__ void k_eval_m2l(interact_item_t *list, int list_size, cell_t *m_cells, cell_t *m_xcells, particle_t *m_particles, particle_t *m_xparticles,
			   iris_real gxsize, iris_real gysize, iris_real gzsize, int m_nterms, iris_real *m_scratch, int m_order, iris_real *m_M, iris_real *m_L)
{
    int tid = IRIS_CUDA_TID;
    if(tid >= list_size) {
	return;
    }
    int srcID = list[tid].sourceID;
    int destID = list[tid].targetID;
    iris_real xoff = list[tid].ix * gxsize;
    iris_real yoff = list[tid].iy * gysize;
    iris_real zoff = list[tid].iz * gzsize;
    
    assert((m_xcells[srcID].flags & IRIS_FMM_CELL_VALID_M));

    iris_real sx = m_xcells[srcID].ses.c.r[0] + xoff;
    iris_real sy = m_xcells[srcID].ses.c.r[1] + yoff;
    iris_real sz = m_xcells[srcID].ses.c.r[2] + zoff;

    iris_real tx = m_cells[destID].ses.c.r[0];
    iris_real ty = m_cells[destID].ses.c.r[1];
    iris_real tz = m_cells[destID].ses.c.r[2];

    iris_real x = tx - sx;
    iris_real y = ty - sy;
    iris_real z = tz - sz;

    memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
    m2l(m_order, x, y, z, m_M + srcID * 2 * m_nterms, m_L + destID * 2 * m_nterms, m_scratch);

    m_cells[destID].flags |= IRIS_FMM_CELL_VALID_L;
}

void fmm::eval_m2l_gpu()
{
    int n = m_m2l_list.size();
    m_logger->info("M2L size = %d", m_m2l_list.size());
    if(n == 0) {
	return;
    }

    m_m2l_list_gpu = (interact_item_t *)memory::wmalloc_gpu_cap(m_m2l_list_gpu, n, sizeof(interact_item_t), &m_m2l_list_cap);
    cudaMemcpyAsync(m_m2l_list_gpu, m_m2l_list.data(), n * sizeof(interact_item_t), cudaMemcpyDefault, m_streams[1]);
    
    int nthreads = IRIS_CUDA_NTHREADS;
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_eval_m2l<<<nblocks, nthreads, 0, m_streams[1]>>>(m_m2l_list_gpu, n, m_cells, m_xcells, m_particles, m_xparticles,
    						       m_domain->m_global_box.xsize, m_domain->m_global_box.ysize, m_domain->m_global_box.zsize, m_nterms, m_scratch,
    						       m_order, m_M, m_L);
}


//////////////
// Eval L2L //
//////////////


__global__ void k_eval_l2l(cell_t *m_cells, int offset, int children_offset, iris_real *m_L, int m_nterms, iris_real *m_scratch, int m_order)
{
    int tid = IRIS_CUDA_TID;
    int scellID = tid + offset;

    if(tid + offset >= children_offset) {
	return;
    }
    
    if(!(m_cells[scellID].flags & IRIS_FMM_CELL_VALID_L)) {
	return;
    }
    
    iris_real cx = m_cells[scellID].ses.c.r[0];
    iris_real cy = m_cells[scellID].ses.c.r[1];
    iris_real cz = m_cells[scellID].ses.c.r[2];

    iris_real *L = m_L + scellID * 2 * m_nterms;
    for(int j=0;j<8;j++) {
	int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
	if(!(m_cells[scellID].flags & mask)) {
	    continue;
	}
	int tcellID = children_offset + 8*tid + j;
	iris_real x = cx - m_cells[tcellID].ses.c.r[0];
	iris_real y = cy - m_cells[tcellID].ses.c.r[1];
	iris_real z = cz - m_cells[tcellID].ses.c.r[2];
	memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
	l2l(m_order, x, y, z, L, m_L + tcellID * 2 * m_nterms, m_scratch);
	m_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_L;
    }
}

void fmm::eval_l2l_gpu()
{
    for(int level = 0; level < m_depth-1; level++) {
	int start = cell_meta_t::offset_for_level(level);
	int end = cell_meta_t::offset_for_level(level+1);
	int n = end - start;
	int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
	int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
	k_eval_l2l<<<nblocks, nthreads>>>(m_cells, start, end, m_L, m_nterms, m_scratch, m_order);
    }
}


//////////////
// Eval L2P //
//////////////


__global__ void k_eval_l2p(cell_t *m_cells, int offset, int end, particle_t *m_particles, int m_order, iris_real *m_L, int m_nterms, iris_real *m_scratch)
{
    int tid = IRIS_CUDA_TID;
    int cellID = tid + offset;
    if(cellID >= end) {
	return;
    }

    cell_t *leaf = m_cells + cellID;
    
    if(leaf->num_children == 0 || !(leaf->flags & IRIS_FMM_CELL_VALID_L)) {
	return;
    }

    iris_real *L = m_L + cellID * 2 * m_nterms;
    for(int j=0;j<leaf->num_children;j++) {
	iris_real x = leaf->ses.c.r[0] - m_particles[leaf->first_child+j].xyzq[0];
	iris_real y = leaf->ses.c.r[1] - m_particles[leaf->first_child+j].xyzq[1];
	iris_real z = leaf->ses.c.r[2] - m_particles[leaf->first_child+j].xyzq[2];
	iris_real q = m_particles[leaf->first_child+j].xyzq[3];

	iris_real phi, Ex, Ey, Ez;
	memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
	l2p(m_order, x, y, z, q, L, m_scratch, &phi, &Ex, &Ey, &Ez);
	
	atomicAdd(m_particles[leaf->first_child+j].tgt + 0, phi);
	atomicAdd(m_particles[leaf->first_child+j].tgt + 1, Ex);
	atomicAdd(m_particles[leaf->first_child+j].tgt + 2, Ey);
	atomicAdd(m_particles[leaf->first_child+j].tgt + 3, Ez);
    }
}

void fmm::eval_l2p_gpu()
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int n = m_tree_size - offset;
    int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_eval_l2p<<<nblocks, nthreads>>>(m_cells, offset, m_tree_size, m_particles, m_order, m_L, m_nterms, m_scratch);    
}


///////////////////////////////
// Compute energy and virial //
///////////////////////////////


// TODO: compute virial
__global__ void k_compute_energy_and_virial(particle_t *m_particles, iris_real *out_ener)
{
    __shared__ iris_real ener_acc[IRIS_CUDA_NTHREADS];
    int iacc = threadIdx.x;
    ener_acc[iacc] = 0.0;
    
    int tid = IRIS_CUDA_TID;
    ener_acc[iacc] += m_particles[tid].tgt[0] * m_particles[tid].xyzq[3];

    __syncthreads();

    for(int i=blockDim.x; i>0; i/=2) {
	int stride = blockDim.x/i;
	if(iacc < (blockDim.x - stride) && iacc % (2*stride) == 0) {
	    ener_acc[iacc] += ener_acc[iacc+stride];
	}
	__syncthreads();
    }
    if(iacc == 0) {
	atomicAdd(out_ener, ener_acc[0]);
    }
}

void fmm::compute_energy_and_virial_gpu()
{
    int n = m_nparticles;
    int nthreads = MIN(IRIS_CUDA_NTHREADS, n);
    int nblocks = IRIS_CUDA_NBLOCKS(n, nthreads);
    k_compute_energy_and_virial<<<nblocks, nthreads>>>(m_particles, m_evir_gpu);
    cudaMemcpy(&(m_iris->m_Ek), m_evir_gpu, sizeof(iris_real), cudaMemcpyDefault);
    
    m_iris->m_Ek *= 0.5 * m_units->ecf;
}

#endif
