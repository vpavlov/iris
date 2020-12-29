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
#include <assert.h>
#include "fmm.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "fmm_kernels.h"
#include "fmm_pair.h"
#include "logger.h"
#include "domain.h"
#include "comm_rec.h"
#include "timer.h"
#include "openmp.h"
#include "proc_grid.h"
#include "tags.h"
#include "ses.h"

#ifdef IRIS_CUDA
#include "cuda_runtime_api.h"
#include "cuda.h"
#endif

using namespace ORG_NCSA_IRIS;

#define _LN8 2.0794415416798357  // natural logarithm of 8

#define MIN_DEPTH 2   // minimum value for depth (2)
#define MAX_DEPTH 16  // more than enough (e.g. 18 quadrillion particles)

fmm::fmm(iris *obj):
    solver(obj), m_order(0), m_depth(0), m_mac(0.0), m_mac_let_corr(0.0), m_nterms(0),
    m_leaf_size{0.0, 0.0, 0.0}, m_local_root_level(0), 
    m_scratch(NULL), m_tree_size(0), m_cell_meta(NULL), m_M(NULL), m_L(NULL),
    m_cells(NULL), m_xcells(NULL), m_nparticles(0), m_npart_cap(0), m_particles(NULL),
    m_nxparticles(0), m_nxpart_cap(0), m_xparticles(NULL),
    m_dirty(true), m_sendcnt(NULL), m_senddisp(NULL), m_recvcnt(NULL), m_recvdisp(NULL),
    m_p2m_count(0), m_m2m_count(0), m_m2l_count(0), m_p2p_count(0),
    m_l2l_count(0), m_l2p_count(0), m_p2m_alien_count(0), m_m2m_alien_count(0),
    m_sendbuf(NULL), m_recvbuf(NULL), m_sendbuf_cap(0), m_recvbuf_cap(0)
#ifdef IRIS_CUDA
    ,m_atom_types(NULL), m_at_cap(0), m_cellID_keys(NULL), m_cellID_keys_cap(0),
    m_cells_cpu(NULL), m_xcells_cpu(NULL), m_M_cpu(NULL), m_recvbuf_gpu(NULL), m_recvbuf_gpu_cap(0),
    m_p2p_list_gpu(NULL), m_p2p_list_cap(0), m_m2l_list_gpu(NULL), m_m2l_list_cap(0)
#endif
{
    int size = sizeof(box_t<iris_real>) * m_iris->m_server_size;
    m_ext_boxes = (box_t<iris_real> *)memory::wmalloc(size);

#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	for(int i=0;i<IRIS_CUDA_FMM_NUM_STREAMS;i++) {
	    cudaStreamCreate(&m_streams[i]);
	}
    }
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);  // otherwise distribute_particles won't work because of the welzl recursion
    IRIS_CUDA_CHECK_ERROR;
#endif
    
}

fmm::~fmm()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	for(int i=0;i<IRIS_CUDA_FMM_NUM_STREAMS;i++) {
	    cudaStreamDestroy(m_streams[i]);
	}
	memory::wfree(m_ext_boxes);
	memory::destroy_1d_gpu(m_scratch);
	memory::destroy_1d_gpu(m_M);
	if(m_M_cpu != NULL) { memory::wfree_gpu(m_M_cpu); };
	memory::destroy_1d_gpu(m_L);
	if(m_cells != NULL) { memory::wfree_gpu(m_cells); }
	if(m_cells_cpu != NULL) { memory::wfree_gpu(m_cells_cpu); }
	if(m_xcells_cpu != NULL) { memory::wfree_gpu(m_xcells_cpu); }
	if(m_xcells != NULL) { memory::wfree_gpu(m_xcells); }
	if(m_particles != NULL) { memory::wfree_gpu(m_particles); }
	// if(m_xparticles != NULL)  { memory::wfree_gpu(m_xparticles); }  // xparticles is at the end of particles
	for(auto it = m_charges_gpu.begin(); it != m_charges_gpu.end(); it++) {
	    memory::wfree_gpu(it->second);
	}
	if(m_atom_types != NULL) { memory::wfree_gpu(m_atom_types); }
	if(m_cellID_keys != NULL) { memory::wfree_gpu(m_cellID_keys); }
	if(m_recvbuf_gpu != NULL) { memory::wfree_gpu(m_recvbuf_gpu); }
	if(m_p2p_list_gpu != NULL) { memory::wfree_gpu(m_p2p_list_gpu); }
	if(m_m2l_list_gpu != NULL) { memory::wfree_gpu(m_m2l_list_gpu); }
    }else 
#endif
    {
	memory::wfree(m_ext_boxes);
	memory::destroy_1d(m_scratch);
	memory::destroy_1d(m_cell_meta);
	memory::destroy_1d(m_M);
	memory::destroy_1d(m_L);
	if(m_cells != NULL) { memory::wfree(m_cells); }
	if(m_xcells != NULL) { memory::wfree(m_xcells); }
	if(m_particles != NULL) { memory::wfree(m_particles); }
	if(m_xparticles != NULL)  { memory::wfree(m_xparticles); }
    }
    if(m_sendcnt != NULL) { memory::wfree(m_sendcnt); }
    if(m_senddisp != NULL) { memory::wfree(m_senddisp); }
    if(m_recvcnt != NULL) { memory::wfree(m_recvcnt); }
    if(m_recvdisp != NULL) { memory::wfree(m_recvdisp); }
    if(m_sendbuf != NULL) { memory::wfree(m_sendbuf); }
    if(m_recvbuf != NULL) { memory::wfree(m_recvbuf); }
}

void fmm::commit()
{
    if(m_dirty) {
	
	m_order = m_iris->m_order;  // if p = 2, we expand multipoles until Y_2^2
	
	int natoms = m_iris->m_natoms;  // atoms won't change during simulation (hopefully)
	solver_param_t t = m_iris->get_solver_param(IRIS_SOLVER_FMM_NCRIT);
	int ncrit = t.i;

	ncrit = MIN(ncrit, IRIS_MAX_NCRIT);
	
	m_depth = (natoms > ncrit) ? int(log(natoms / ncrit)/_LN8) + 2 : 0;
	m_depth = MAX(m_depth, MIN_DEPTH);
	m_depth = MIN(m_depth, MAX_DEPTH);

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC);
	m_mac = t.r;

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC_CORR);
	m_mac_let_corr = t.r;
	
	m_nterms = (m_order + 1) * (m_order + 2) / 2;

	m_local_root_level = int(log(m_local_comm->m_size-1) / _LN8) + 1;
	if(m_local_comm->m_size == 1) {
	    m_local_root_level = 0;
	}

	m_tree_size = ((1 << 3 * m_depth) - 1) / 7;
	
	handle_box_resize();

#ifdef IRIS_CUDA
	if(m_iris->m_cuda) {
	    memory::destroy_1d_gpu(m_scratch);
	    memory::create_1d_gpu(m_scratch, 2*m_nterms);
	    memory::destroy_1d_gpu(m_M);
	    memory::create_1d_gpu(m_M, m_tree_size*2*m_nterms);
	    if(m_M_cpu != NULL) { memory::wfree_gpu(m_M_cpu); };
	    m_M_cpu = (iris_real *)memory::wmalloc_gpu(m_tree_size*2*m_nterms*sizeof(iris_real), false, true);
	    memory::destroy_1d_gpu(m_L);
	    memory::create_1d_gpu(m_L, m_tree_size*2*m_nterms);

	    if(m_cells != NULL) { memory::wfree_gpu(m_cells); }
	    m_cells = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t));

	    if(m_cells_cpu != NULL) { memory::wfree_gpu(m_cells_cpu); }
	    m_cells_cpu = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t), false, true);

	    if(m_xcells_cpu != NULL) { memory::wfree_gpu(m_xcells_cpu); }
	    m_xcells_cpu = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t), false, true);
	    
	    if(m_xcells != NULL) { memory::wfree_gpu(m_xcells); }
	    m_xcells = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t));
	    
	}else
#endif
	{
	    memory::destroy_1d(m_scratch);
	    memory::create_1d(m_scratch, 2*m_nterms);
	    memory::destroy_1d(m_M);
	    memory::create_1d(m_M, m_tree_size*2*m_nterms);
	    memory::destroy_1d(m_L);
	    memory::create_1d(m_L, m_tree_size*2*m_nterms);
	    
	    memory::destroy_1d(m_cells);
	    memory::create_1d(m_cells, m_tree_size);
	    
	    memory::destroy_1d(m_xcells);
	    memory::create_1d(m_xcells, m_tree_size);
	}

	memory::destroy_1d(m_sendcnt);
	memory::create_1d(m_sendcnt, m_local_comm->m_size);
	
	memory::destroy_1d(m_senddisp);
	memory::create_1d(m_senddisp, m_local_comm->m_size);
	
	memory::destroy_1d(m_recvcnt);
	memory::create_1d(m_recvcnt, m_local_comm->m_size);
	
	memory::destroy_1d(m_recvdisp);
	memory::create_1d(m_recvdisp, m_local_comm->m_size);
	
	m_dirty = false;
	if(m_iris->m_cuda) {
	    m_logger->info("FMM/GPU: order = %d; depth = %d; tree size = %d; local root level = %d", m_order, m_depth, m_tree_size, m_local_root_level);
	}else {
	    m_logger->info("FMM: order = %d; depth = %d; tree size = %d; local root level = %d", m_order, m_depth, m_tree_size, m_local_root_level);
	}
    }
    calc_ext_boxes();  // this is needed every time
}

void fmm::generate_cell_meta()
{
    memory::destroy_1d(m_cell_meta);
    memory::create_1d(m_cell_meta, m_tree_size);
    for(int i=0;i<m_tree_size;i++) {
	m_cell_meta[i].set(m_cell_meta, i, &m_domain->m_global_box, m_leaf_size, max_level(), m_local_comm->m_size, m_local_root_level);
    }
}

void fmm::set_leaf_size()
{
    int nd = 1 << max_level();

    m_leaf_size[0] = m_domain->m_global_box.xsize / nd;
    m_leaf_size[1] = m_domain->m_global_box.ysize / nd;
    m_leaf_size[2] = m_domain->m_global_box.zsize / nd;
}

void fmm::handle_box_resize()
{
    set_leaf_size();
    // Cell meta is no longer needed, except for one-sided comm, which is not implemented yet (maybe never)
    generate_cell_meta();  
}

void fmm::compute_energy_and_virial()
{
    iris_real ener = 0.0;
    for(int i=0;i<m_nparticles;i++) {
    	ener += m_particles[i].tgt[0] * m_particles[i].xyzq[3];
    }
    m_iris->m_Ek = ener * 0.5 * m_units->ecf;
    // TODO: calculate virial in m_iris->m_virial[0..5]
}

void fmm::send_forces_to(int peer, int start, int end, bool include_energy_virial)
{
    MPI_Comm comm = m_iris->client_comm();
    int ncharges = end - start;
    int size = 7*sizeof(iris_real) + ncharges*4*sizeof(iris_real);
    iris_real *forces = (iris_real *)memory::wmalloc(size);
    
    if(include_energy_virial) {
	forces[0] = m_iris->m_Ek;
	forces[1] = m_iris->m_virial[0];
	forces[2] = m_iris->m_virial[1];
	forces[3] = m_iris->m_virial[2];
	forces[4] = m_iris->m_virial[3];
	forces[5] = m_iris->m_virial[4];
	forces[6] = m_iris->m_virial[5];
    }else {
	forces[0] = 0.0;
	forces[1] = 0.0;
	forces[2] = 0.0;
	forces[3] = 0.0;
	forces[4] = 0.0;
	forces[5] = 0.0;
	forces[6] = 0.0;
    }
    
    m_iris->m_forces[peer] = forces;

    for(int i=start;i<end;i++) {
	iris_real factor = m_particles[i].xyzq[3] * m_units->ecf;  // q * 1/4pieps

	forces[7 + (i-start)*4 + 0] = m_particles[i].index + 0.33333333;
	forces[7 + (i-start)*4 + 1] = factor * m_particles[i].tgt[1];
	forces[7 + (i-start)*4 + 2] = factor * m_particles[i].tgt[2];
	forces[7 + (i-start)*4 + 3] = factor * m_particles[i].tgt[3];
    }

    MPI_Request req;
    m_iris->send_event(comm, peer, IRIS_TAG_FORCES, size, forces, &req, NULL);
    MPI_Request_free(&req);
}

void fmm::send_back_forces()
{
    bool include_energy_virial = true;  // send the energy and virial to only one of the clients; to the others send 0
    
    sort_back_particles(m_particles, m_nparticles);
    int start = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++) {
	int end;
	for(end = start ; end<m_nparticles ; end++) {
	    if(m_particles[end].rank != rank) {
		break;
	    }
	}
	send_forces_to(rank, start, end, include_energy_virial);
	include_energy_virial = false;
	start = end;
    }
}

void fmm::solve()
{
    timer tm;
    m_logger->trace("FMM solve() start");

    tm.start();
    
    if(m_iris->m_compute_global_energy) {
	m_iris->m_Ek = 0.0;
    }

    if(m_iris->m_compute_global_virial) {
	m_iris->m_virial[0] =
	    m_iris->m_virial[1] =
	    m_iris->m_virial[2] =
	    m_iris->m_virial[3] =
	    m_iris->m_virial[4] =
	    m_iris->m_virial[5] = 0.0;
    }

    m_p2p_list.clear();
    m_m2l_list.clear();
    m_has_cells_cpu = false;
    
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
       	cudaMemsetAsync(m_M, 0, m_tree_size*2*m_nterms*sizeof(iris_real), m_streams[0]);
	cudaMemsetAsync(m_L, 0, m_tree_size*2*m_nterms*sizeof(iris_real), m_streams[1]);
	cudaMemsetAsync(m_cells, 0, m_tree_size*sizeof(cell_t), m_streams[2]);	
    }else
#endif
    {
	memset(m_M, 0, m_tree_size*2*m_nterms*sizeof(iris_real));
	memset(m_L, 0, m_tree_size*2*m_nterms*sizeof(iris_real));
	memset(m_cells, 0, m_tree_size*sizeof(cell_t));
    }
    
    m_p2m_count = m_m2m_count = m_m2l_count = m_p2p_count = m_l2l_count = m_l2p_count = m_p2m_alien_count = m_m2m_alien_count = 0;

    local_tree_construction();
    exchange_LET();    
    dual_tree_traversal();

    if(m_iris->m_cuda) {
#ifdef IRIS_CUDA
	if(m_iris->m_cuda) {
	    cudaDeviceSynchronize();
	    IRIS_CUDA_CHECK_ERROR;
	}
#endif
	MPI_Barrier(m_local_comm->m_comm);
	exit(-1);
    }

    compute_energy_and_virial();
    send_back_forces();

    tm.stop();
    m_logger->info("FMM: Total step wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
    
    m_logger->info("P2M: %d (%d), M2M: %d (%d), M2L: %d, P2P: %d, L2L: %d, L2P: %d", m_p2m_count, m_p2m_alien_count, m_m2m_count, m_m2m_alien_count, m_m2l_count, m_p2p_count, m_l2l_count, m_l2p_count);
}

void fmm::local_tree_construction()
{
    timer tm;
    tm.start();

    load_particles();                                          // creates and sorts the m_particles array
    distribute_particles(m_particles, m_nparticles, IRIS_FMM_CELL_LOCAL, m_cells);  // distribute particles into leaf cells
    link_parents(m_cells);                                     // link parents and calculate parent's SES
    eval_p2m(m_cells, false);                                  // eval P2M for leaf nodes
    eval_m2m(m_cells, false);                                  // eval M2M for non-leaf nodes
    
    tm.stop();
    m_logger->info("FMM: Local tree construction wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
    //print_tree("Cell", m_cells, 0);
}

void fmm::load_particles()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	load_particles_gpu();
    }else 
#endif
    {
	load_particles_cpu();
    }
}

void fmm::load_particles_cpu()
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int nd = 1 << max_level();

    m_nparticles = m_iris->num_local_atoms();
    m_nxparticles = m_iris->num_halo_atoms();
    m_logger->info("FMM: This rank owns %d + %d halo particles", m_nparticles, m_nxparticles);

    m_particles = (particle_t *)memory::wmalloc_cap((void *)m_particles, m_nparticles, sizeof(particle_t), &m_npart_cap);
    m_xparticles = (particle_t *)memory::wmalloc_cap((void *)m_xparticles, m_nxparticles, sizeof(particle_t), &m_nxpart_cap);
        
    box_t<iris_real> *gbox = &m_domain->m_global_box;
    
    int n_own = 0;
    int n_halo = 0;
    int lc[3];  // leaf global coords
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	int ncharges = m_iris->m_ncharges[rank];
	iris_real *charges = m_iris->m_charges[rank];
	for(int i=0;i<ncharges;i++) {
	    iris_real tx = (charges[i * 5 + 0] - gbox->xlo) / m_leaf_size[0];
	    iris_real ty = (charges[i * 5 + 1] - gbox->ylo) / m_leaf_size[1];
	    iris_real tz = (charges[i * 5 + 2] - gbox->zlo) / m_leaf_size[2];

	    lc[0] = (int) tx;
	    lc[1] = (int) ty;
	    lc[2] = (int) tz;

	    int id = 0;
	    for(int l=0;l<max_level(); l++) {
		for(int d=0;d<3;d++) {
		    id += (lc[d] & 1) << (3*l + d);
		    lc[d] >>= 1;
		}
	    }

	    int cellID = offset + id;
	    int chargeID = (int)charges[i*5 + 4];
	    assert(chargeID != 0);
	    
	    if(chargeID > 0) {
		m_particles[n_own].rank = rank;
		m_particles[n_own].index = chargeID;
		m_particles[n_own].cellID = cellID;
		memcpy(m_particles[n_own].xyzq, charges+i*5, 4*sizeof(iris_real));
		memset(m_particles[n_own].tgt, 0, 4*sizeof(iris_real));
		n_own++;
	    }else {
		m_xparticles[n_halo].rank = rank;
		m_xparticles[n_halo].index = -chargeID;
		m_xparticles[n_halo].cellID = cellID;
		memcpy(m_xparticles[n_halo].xyzq, charges+i*5, 4*sizeof(iris_real));
		memset(m_xparticles[n_halo].tgt, 0, 4*sizeof(iris_real));
		n_halo++;
	    }
	}
    }

    // sort the final lists by cellID
    sort_particles(m_particles, m_nparticles, false);
    sort_particles(m_xparticles, m_nxparticles, false);
}

void fmm::distribute_particles(particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	distribute_particles_gpu(in_particles, in_count, in_flags, out_target);
    }else 
#endif
    {
	distribute_particles_cpu(in_particles, in_count, in_flags, out_target);
    }
}

// TODO: implement openmp version of this similar to the CUDA version
void fmm::distribute_particles_cpu(particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    if(in_count == 0) {
	return;
    }
    
    box_t<iris_real> *gbox = &m_domain->m_global_box;
    
    int last = in_particles[0].cellID;
    int first_child = 0;
    int num_children = 0;
    
    for(int i=0;i<in_count;i++) {
	if(in_particles[i].cellID != last) {
	    assert(out_target[last].num_children == 0);  // no case in which two leafs overlap
	    out_target[last].first_child = first_child;
	    out_target[last].num_children = num_children;
	    out_target[last].flags = in_flags;

	    // out_target[last].ses.c.r[0] = m_cell_meta[last].geomc[0];
	    // out_target[last].ses.c.r[1] = m_cell_meta[last].geomc[1];
	    // out_target[last].ses.c.r[2] = m_cell_meta[last].geomc[2];
	    // out_target[last].ses.r = m_cell_meta[last].maxr;
	    out_target[last].compute_ses(in_particles);
	    
	    first_child = i;
	    num_children = 0;
	    last = in_particles[i].cellID;
	}
	num_children++;
    }
    out_target[last].first_child = first_child;
    out_target[last].num_children = num_children;
    out_target[last].flags = in_flags;
    
    // out_target[last].ses.c.r[0] = m_cell_meta[last].geomc[0];
    // out_target[last].ses.c.r[1] = m_cell_meta[last].geomc[1];
    // out_target[last].ses.c.r[2] = m_cell_meta[last].geomc[2];
    // out_target[last].ses.r = m_cell_meta[last].maxr;
    out_target[last].compute_ses(in_particles);
};

void fmm::relink_parents(cell_t *io_cells)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	relink_parents_gpu(io_cells);
    }else
#endif
    {
	relink_parents_cpu(io_cells);
    }
}

void fmm::relink_parents_cpu(cell_t *io_cells)
{
    // first, clear the num_children and ses of all non-leaf cells
    int end = cell_meta_t::offset_for_level(max_level());
    for(int i=0;i<end;i++) {
	io_cells[i].flags &= ~IRIS_FMM_CELL_HAS_CHILDREN;
    }

    // second, for all shared cells (above local root level) recalculate ses
    end = cell_meta_t::offset_for_level(m_local_root_level);
    for(int i=0;i<end;i++) {
	io_cells[i].ses.r = 0.0;
    }

    // now, link all leafs upwards
    link_parents(io_cells);
}

void fmm::link_parents(cell_t *io_cells)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	link_parents_gpu(io_cells);
    }else
#endif
    {
	link_parents_cpu(io_cells);
    }
}

void fmm::link_parents_cpu(cell_t *io_cells)
{
    for(int i=max_level();i>0;i--) {
	int start = cell_meta_t::offset_for_level(i);
	int end = cell_meta_t::offset_for_level(i+1);
	for(int j=start;j<end;j++) {
	    if((io_cells[j].num_children != 0) ||                   // cell is a non-empty leaf
	       (io_cells[j].flags & IRIS_FMM_CELL_HAS_CHILDREN) ||  // or cell is a non-leaf and has some children
	       (io_cells[j].flags & IRIS_FMM_CELL_ALIEN_NL)) {        // or is an alien cell
		int parent = cell_meta_t::parent_of(j);
		io_cells[parent].flags |= (IRIS_FMM_CELL_HAS_CHILD1 << ((j - start) % 8));
	    }
	}
    }

    for(int i=max_level()-1;i>=0;i--) {
	int start = cell_meta_t::offset_for_level(i);
	int end = cell_meta_t::offset_for_level(i+1);
	for(int j=start;j<end;j++) {
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
}

void fmm::eval_p2m(cell_t *in_cells, bool alien_only)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_p2m_gpu(in_cells, alien_only);
    }else
#endif
    {
	eval_p2m_cpu(in_cells, alien_only);
    }
}

void fmm::eval_p2m_cpu(cell_t *in_cells, bool alien_only)
{
    int offset = cell_meta_t::offset_for_level(max_level());
    for(int i=offset;i<m_tree_size;i++) {
	cell_t *leaf = &in_cells[i];
	
	// no particles here -- continue
	if(leaf->num_children == 0) {
	    continue;
	}
	
	// we only want alien cells, but this one is local -- continue
	if(alien_only && !(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF)) {
	    continue;
	}

	// it has been send from exchange_LET AND from halo exchange -- continue
	if(alien_only && (leaf->flags & IRIS_FMM_CELL_ALIEN_NL)) {
	    continue;
	}
	for(int j=0;j<leaf->num_children;j++) {
	    iris_real x, y, z, q;
	    if(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		x = m_xparticles[leaf->first_child+j].xyzq[0] - in_cells[i].ses.c.r[0];
		y = m_xparticles[leaf->first_child+j].xyzq[1] - in_cells[i].ses.c.r[1];
		z = m_xparticles[leaf->first_child+j].xyzq[2] - in_cells[i].ses.c.r[2];
		q = m_xparticles[leaf->first_child+j].xyzq[3];
	    }else {
		x = m_particles[leaf->first_child+j].xyzq[0] - in_cells[i].ses.c.r[0];
		y = m_particles[leaf->first_child+j].xyzq[1] - in_cells[i].ses.c.r[1];
		z = m_particles[leaf->first_child+j].xyzq[2] - in_cells[i].ses.c.r[2];
		q = m_particles[leaf->first_child+j].xyzq[3];
	    }
	    p2m(m_order, x, y, z, q, m_M + i * 2 * m_nterms);
	    in_cells[i].flags |= IRIS_FMM_CELL_VALID_M;
	    m_p2m_count++;
	    if(alien_only) {
		m_p2m_alien_count++;
	    }
	}
    }
}

void fmm::eval_m2m(cell_t *in_cells, bool invalid_only)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_m2m_gpu(in_cells, invalid_only);
    }else
#endif
    {
	eval_m2m_cpu(in_cells, invalid_only);
    }
}

void fmm::eval_m2m_cpu(cell_t *in_cells, bool invalid_only)
{
    int last_level = invalid_only ? 0 : m_local_root_level;
    for(int level = max_level()-1;level>=last_level;level--) {
	int tcellID = cell_meta_t::offset_for_level(level);
	int scellID = cell_meta_t::offset_for_level(level+1);
	int ntcells = scellID - tcellID;
	for(int i = 0;i<ntcells;i++) {
	    if(invalid_only && (in_cells[tcellID].flags & IRIS_FMM_CELL_VALID_M)) {
		tcellID++;
		scellID+=8;
		continue;
	    }
	    
	    iris_real cx = in_cells[tcellID].ses.c.r[0];
	    iris_real cy = in_cells[tcellID].ses.c.r[1];
	    iris_real cz = in_cells[tcellID].ses.c.r[2];

	    bool valid_m = false;
	    for(int j=0;j<8;j++) {
		int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
		if(!(in_cells[tcellID].flags & mask)) {
		    scellID++;
		    continue;
		}
		iris_real x = in_cells[scellID].ses.c.r[0] - cx;
		iris_real y = in_cells[scellID].ses.c.r[1] - cy;
		iris_real z = in_cells[scellID].ses.c.r[2] - cz;
		memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
		m2m(m_order, x, y, z, m_M + scellID * 2 * m_nterms, m_M + tcellID * 2 * m_nterms, m_scratch);
		valid_m = true;
		scellID++;
		m_m2m_count++;
		if(invalid_only) {
		    m_m2m_alien_count++;
		}
	    }
	    if(valid_m) {
		in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_M;
	    }
	    tcellID++;
	}
    }
}

void fmm::exchange_LET()
{
    timer tm, tm3;
    tm.start();

#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	cudaMemcpy(m_xcells, m_cells, m_tree_size * sizeof(cell_t), cudaMemcpyDefault);  // copy local tree to LET
    }else
#endif
    {
	memcpy(m_xcells, m_cells, m_tree_size * sizeof(cell_t));  // copy local tree to LET
    }
    
    if(m_local_comm->m_size > 1) {
	distribute_particles(m_xparticles, m_nxparticles, IRIS_FMM_CELL_ALIEN_LEAF, m_xcells);  // distribute particles into leaf cells
	comm_LET();
	recalculate_LET();
    }
    //print_tree("Xcell", m_xcells, 0);
    
    tm.stop();
    m_logger->info("FMM: Exchange LET Total wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());    
}

void fmm::comm_LET()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	int count = comm_LET_gpu();
	inhale_xcells_gpu(count);
    }else
#endif
    {
	int count = comm_LET_cpu(m_cells, m_M);
	inhale_xcells(count);
    }
}

void fmm::recalculate_LET()
{
    relink_parents(m_xcells);
    eval_p2m(m_xcells, true);
    eval_m2m(m_xcells, true);
}

#ifdef IRIS_CUDA
void fmm::print_tree_gpu(const char *label, cell_t *in_cells)
{
    cell_t *tmp = (cell_t *)memory::wmalloc(m_tree_size * sizeof(cell_t));
    cudaMemcpy(tmp, in_cells, m_tree_size * sizeof(cell_t), cudaMemcpyDefault);
    print_tree(label, tmp, 0);
    memory::wfree(tmp);
}
#endif

void fmm::print_tree(const char *label, cell_t *in_cells, int cellID)
{
    int level = cell_meta_t::level_of(cellID);
    if(level == max_level()) {
	m_logger->info("%*s%s %d (L%d) N=%d F=0x%x C=(%f,%f,%f) R=%f M[0] = %f", level+1, " ", label, cellID, level, in_cells[cellID].num_children, in_cells[cellID].flags,
		       in_cells[cellID].ses.c.r[0],
		       in_cells[cellID].ses.c.r[1],
		       in_cells[cellID].ses.c.r[2],
		       in_cells[cellID].ses.r,
		       m_M[cellID*2*m_nterms]);
    }else {
	int num_children = 0;
	int mask = IRIS_FMM_CELL_HAS_CHILD1;
	for(int i=0;i<8;i++) {
	    if(in_cells[cellID].flags & mask) {
		num_children++;
	    }
	    mask <<= 1;
	}
	
	m_logger->info("%*s%s %d (L%d) N=%d F=0x%x C=(%f,%f,%f) R=%f M[0] = %f", level+1, " ", label, cellID, level, num_children, in_cells[cellID].flags,
		       in_cells[cellID].ses.c.r[0],
		       in_cells[cellID].ses.c.r[1],
		       in_cells[cellID].ses.c.r[2],
		       in_cells[cellID].ses.r,
		       m_M[cellID*2*m_nterms]);
    }
    if(level < max_level()) {
	int this_offset = cell_meta_t::offset_for_level(level);
	int children_offset = cell_meta_t::offset_for_level(level+1);
	for(int j=0;j<8;j++) {
	    int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
	    if(in_cells[cellID].flags & mask) {
		int childID = children_offset + 8*(cellID-this_offset) + j;
		print_tree(label, in_cells, childID);
	    }
	}
    }
}

void fmm::dual_tree_traversal()
{
    timer tm;
    tm.start();
            
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	dual_tree_traversal_gpu();
	eval_p2p_gpu();
	eval_m2l_gpu();
	// eval_l2l();
	// eval_l2p();
    }else
#endif
    {
	dual_tree_traversal_cpu(m_xcells, m_cells);
	eval_p2p_cpu();
	eval_m2l_cpu();
	eval_l2l();
	eval_l2p();
    }

    tm.stop();
    m_logger->info("FMM: Dual Tree Traversal wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::dual_tree_traversal_gpu()
{
    if(!m_has_cells_cpu) {
	cudaMemcpyAsync(m_cells_cpu, m_cells, m_tree_size*sizeof(cell_t), cudaMemcpyDefault, m_streams[1]);
    }
    cudaMemcpyAsync(m_xcells_cpu, m_xcells, m_tree_size*sizeof(cell_t), cudaMemcpyDefault, m_streams[1]);
    cudaStreamSynchronize(m_streams[1]);
    dual_tree_traversal_cpu(m_xcells_cpu, m_cells_cpu);
}

void fmm::dual_tree_traversal_cpu(cell_t *src_cells, cell_t *dest_cells)
{
    assert(m_queue.empty());
    
    for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
	    for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		pair_t root;
		root.sourceID = 0;
		root.targetID = 0;
		m_queue.push_back(root);
		traverse_queue(src_cells, dest_cells, ix, iy, iz);
	    }
	}
    }
}

void fmm::traverse_queue(cell_t *src_cells, cell_t *dest_cells, int ix, int iy, int iz)
{
    while(!m_queue.empty()) {
	pair_t pair = m_queue.front();
	m_queue.pop_front();

	int src_level = cell_meta_t::level_of(pair.sourceID);
	int tgt_level = cell_meta_t::level_of(pair.targetID);
	
	if((tgt_level == max_level()) || (src_level != max_level() && src_cells[pair.sourceID].ses.r > dest_cells[pair.targetID].ses.r)) {
	    cell_t *src = src_cells + pair.sourceID;
	    int level = cell_meta_t::level_of(pair.sourceID);
	    int this_offset = cell_meta_t::offset_for_level(level);
	    int children_offset = cell_meta_t::offset_for_level(level+1);
	    int mask = IRIS_FMM_CELL_HAS_CHILD1;
	    for(int i=0;i<8;i++) {
		if(src_cells[pair.sourceID].flags & mask) {
		    int childID = children_offset + (pair.sourceID - this_offset)*8 + i;
		    interact(src_cells, dest_cells, childID, pair.targetID, ix, iy, iz);
		}
		mask <<= 1;
	    }
	}else {
	    cell_t *target = dest_cells + pair.targetID;
	    int level = cell_meta_t::level_of(pair.targetID);
	    int this_offset = cell_meta_t::offset_for_level(level);
	    int children_offset = cell_meta_t::offset_for_level(level+1);
	    int mask = IRIS_FMM_CELL_HAS_CHILD1;
	    for(int i=0;i<8;i++) {
		if(dest_cells[pair.targetID].flags & mask) {
		    int childID = children_offset + (pair.targetID - this_offset)*8 + i;
		    interact(src_cells, dest_cells, pair.sourceID, childID, ix, iy, iz);
		}
		mask <<= 1;
	    }
	}
    }
}

void fmm::interact(cell_t *src_cells, cell_t *dest_cells, int srcID, int destID, int ix, int iy, int iz)
{
    iris_real src_cx = src_cells[srcID].ses.c.r[0] + ix * m_domain->m_global_box.xsize;
    iris_real src_cy = src_cells[srcID].ses.c.r[1] + iy * m_domain->m_global_box.ysize;
    iris_real src_cz = src_cells[srcID].ses.c.r[2] + iz * m_domain->m_global_box.zsize;

    iris_real dest_cx = dest_cells[destID].ses.c.r[0];
    iris_real dest_cy = dest_cells[destID].ses.c.r[1];
    iris_real dest_cz = dest_cells[destID].ses.c.r[2];

    iris_real dx = dest_cx - src_cx;
    iris_real dy = dest_cy - src_cy;
    iris_real dz = dest_cz - src_cz;
    
    iris_real rn = sqrt(dx*dx + dy*dy + dz*dz);
    iris_real dn = src_cells[srcID].ses.r + dest_cells[destID].ses.r;
    if(dn/rn < m_mac) {
	interact_item_t t(srcID, destID, ix, iy, iz);
	m_m2l_list.push_back(t);
    }else if(cell_meta_t::level_of(srcID) == max_level() &&
	     cell_meta_t::level_of(destID) == max_level())
    {
	interact_item_t t(srcID, destID, ix, iy, iz);
	m_p2p_list.push_back(t);
    }else {
	pair_t pair;
	pair.sourceID = srcID;
	pair.targetID = destID;
	m_queue.push_back(pair);
    }
}

// TODO: OpenMP
void fmm::eval_p2p_cpu()
{
    for(int i=0;i<m_p2p_list.size();i++) {
	interact_item_t *item = &(m_p2p_list[i]);
	eval_p2p(item->sourceID, item->targetID, item->ix, item->iy, item->iz);
    }
}

// TODO: OpenMP
void fmm::eval_m2l_cpu()
{
    for(int i=0;i<m_m2l_list.size();i++) {
	interact_item_t *item = &(m_m2l_list[i]);
	eval_m2l(item->sourceID, item->targetID, item->ix, item->iy, item->iz);
    }
}

void fmm::eval_p2p(int srcID, int destID, int ix, int iy, int iz)
{
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
		sx = m_xparticles[m_xcells[srcID].first_child + j].xyzq[0] + ix * m_domain->m_global_box.xsize;
		sy = m_xparticles[m_xcells[srcID].first_child + j].xyzq[1] + iy * m_domain->m_global_box.ysize;
		sz = m_xparticles[m_xcells[srcID].first_child + j].xyzq[2] + iz * m_domain->m_global_box.zsize;
		sq = m_xparticles[m_xcells[srcID].first_child + j].xyzq[3];
	    }else {
		sx = m_particles[m_xcells[srcID].first_child + j].xyzq[0] + ix * m_domain->m_global_box.xsize;
		sy = m_particles[m_xcells[srcID].first_child + j].xyzq[1] + iy * m_domain->m_global_box.ysize;
		sz = m_particles[m_xcells[srcID].first_child + j].xyzq[2] + iz * m_domain->m_global_box.zsize;
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

	m_particles[m_cells[destID].first_child + i].tgt[0] += sum_phi;
	m_particles[m_cells[destID].first_child + i].tgt[1] += sum_ex;
	m_particles[m_cells[destID].first_child + i].tgt[2] += sum_ey;
	m_particles[m_cells[destID].first_child + i].tgt[3] += sum_ez;
    }
    m_p2p_count++;
}

void fmm::eval_m2l(int srcID, int destID, int ix, int iy, int iz)
{
    assert((m_xcells[srcID].flags & IRIS_FMM_CELL_VALID_M));

    iris_real sx = m_xcells[srcID].ses.c.r[0] + ix * m_domain->m_global_box.xsize;
    iris_real sy = m_xcells[srcID].ses.c.r[1] + iy * m_domain->m_global_box.ysize;
    iris_real sz = m_xcells[srcID].ses.c.r[2] + iz * m_domain->m_global_box.zsize;

    iris_real tx = m_cells[destID].ses.c.r[0];
    iris_real ty = m_cells[destID].ses.c.r[1];
    iris_real tz = m_cells[destID].ses.c.r[2];

    iris_real x = tx - sx;
    iris_real y = ty - sy;
    iris_real z = tz - sz;

    memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
    m2l(m_order, x, y, z, m_M + srcID * 2 * m_nterms, m_L + destID * 2 * m_nterms, m_scratch);

    m_cells[destID].flags |= IRIS_FMM_CELL_VALID_L;
    m_m2l_count++;
}

void fmm::eval_l2l()
{
    for(int level = 0; level < m_depth-1; level++) {
	int scellID = cell_meta_t::offset_for_level(level);
	int tcellID = cell_meta_t::offset_for_level(level+1);
	int nscells = tcellID - scellID;
	for(int i = 0;i<nscells;i++) {
	    if(!(m_cells[scellID].flags & IRIS_FMM_CELL_VALID_L)) {
		scellID++;
		tcellID += 8;
		continue;
	    }
	    
	    iris_real cx = m_cells[scellID].ses.c.r[0];
	    iris_real cy = m_cells[scellID].ses.c.r[1];
	    iris_real cz = m_cells[scellID].ses.c.r[2];

	    for(int j=0;j<8;j++) {
		int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
		if(!(m_cells[scellID].flags & mask)) {
		    tcellID++;
		    continue;
		}
		iris_real x = cx - m_cells[tcellID].ses.c.r[0];
		iris_real y = cy - m_cells[tcellID].ses.c.r[1];
		iris_real z = cz - m_cells[tcellID].ses.c.r[2];

		memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
		l2l(m_order, x, y, z, m_L + scellID * 2 * m_nterms, m_L + tcellID * 2 * m_nterms, m_scratch);
		m_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_L;
		tcellID++;
		m_l2l_count++;
	    }
	    scellID++;
	}
    }
}

void fmm::eval_l2p()
{
    int offset = cell_meta_t::offset_for_level(max_level());
    for(int i=offset;i<m_tree_size;i++) {
    	cell_t *leaf = m_cells + i;
    	if(leaf->num_children == 0 || !(leaf->flags & IRIS_FMM_CELL_VALID_L)) {
    	    continue;
    	}
    	for(int j=0;j<leaf->num_children;j++) {
	    iris_real x = leaf->ses.c.r[0] - m_particles[leaf->first_child+j].xyzq[0];
	    iris_real y = leaf->ses.c.r[1] - m_particles[leaf->first_child+j].xyzq[1];
	    iris_real z = leaf->ses.c.r[2] - m_particles[leaf->first_child+j].xyzq[2];
	    iris_real q = m_particles[leaf->first_child+j].xyzq[3];
	    iris_real phi, Ex, Ey, Ez;
	    
	    memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
	    l2p(m_order, x, y, z, q, m_L + i * 2 * m_nterms, m_scratch, &phi, &Ex, &Ey, &Ez);
	    
	    m_particles[leaf->first_child+j].tgt[0] += phi;
 	    m_particles[leaf->first_child+j].tgt[1] += Ex;
	    m_particles[leaf->first_child+j].tgt[2] += Ey;
	    m_particles[leaf->first_child+j].tgt[3] += Ez;
	    
	    m_l2p_count++;
    	}
    }
}

void fmm::calc_ext_boxes()
{
    iris_real R = sqrt(m_leaf_size[0]*m_leaf_size[0] + m_leaf_size[1]*m_leaf_size[1] + m_leaf_size[2]*m_leaf_size[2]);

    iris_real r_cut = 2 * R / m_mac;
    int nx = (int)(r_cut / m_leaf_size[0]);
    int ny = (int)(r_cut / m_leaf_size[1]);
    int nz = (int)(r_cut / m_leaf_size[2]);

    m_ext_box.xlo = m_domain->m_local_box.xlo - nx * m_leaf_size[0];
    m_ext_box.ylo = m_domain->m_local_box.ylo - ny * m_leaf_size[1];
    m_ext_box.zlo = m_domain->m_local_box.zlo - nz * m_leaf_size[2];
    
    m_ext_box.xhi = m_domain->m_local_box.xhi + nx * m_leaf_size[0];
    m_ext_box.yhi = m_domain->m_local_box.yhi + ny * m_leaf_size[1];
    m_ext_box.zhi = m_domain->m_local_box.zhi + nz * m_leaf_size[2];
    
    if(m_proc_grid->m_pbc[0] == 0) {
	m_ext_box.xlo = MAX(m_ext_box.xlo, 0.0); 
	m_ext_box.xhi = MIN(m_ext_box.xhi, m_domain->m_global_box.xhi);
    }
    
    if(m_proc_grid->m_pbc[1] == 0) {
	m_ext_box.ylo = MAX(m_ext_box.ylo, 0.0);
	m_ext_box.yhi = MIN(m_ext_box.yhi, m_domain->m_global_box.yhi);
    }

    if(m_proc_grid->m_pbc[2] == 0) {
	m_ext_box.zlo = MAX(m_ext_box.zlo, 0.0);
	m_ext_box.zhi = MIN(m_ext_box.zhi, m_domain->m_global_box.zhi);
    }

    
    m_ext_box.xsize = m_ext_box.xhi - m_ext_box.xlo;
    m_ext_box.ysize = m_ext_box.yhi - m_ext_box.ylo;
    m_ext_box.zsize = m_ext_box.zhi - m_ext_box.zlo;
    
    MPI_Allgather(&m_ext_box, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_ext_boxes, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_local_comm->m_comm);

    m_logger->info("Extended box is %g x %g x %g: [%g:%g][%g:%g][%g:%g]",
		   m_ext_box.xsize, m_ext_box.ysize,
		   m_ext_box.zsize,
		   m_ext_box.xlo, m_ext_box.xhi,
		   m_ext_box.ylo, m_ext_box.yhi,
		   m_ext_box.zlo, m_ext_box.zhi);
}

box_t<iris_real> *fmm::get_ext_boxes()
{
    return m_ext_boxes;
}
