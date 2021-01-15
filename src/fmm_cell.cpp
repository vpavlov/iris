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
#include <cmath>
#include "cuda.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "ses.h"

using namespace ORG_NCSA_IRIS;

//
// Determine the center of the cell by using its cellID, the global box and the leaf size
//
void cell_meta_t::set(cell_meta_t *in_meta, int cellID, const box_t<iris_real> *in_gbox, iris_real *in_leaf_size, int in_max_level,
		      int in_comm_size, int in_local_root_level)
{
    int level = level_of(cellID);
    
    if(cellID == 0) {
	geomc[0] = (in_gbox->xlo + in_gbox->xhi)/2;
	geomc[1] = (in_gbox->ylo + in_gbox->yhi)/2;
	geomc[2] = (in_gbox->zlo + in_gbox->zhi)/2;
	maxr = sqrt(in_gbox->xsize*in_gbox->xsize + in_gbox->ysize*in_gbox->ysize + in_gbox->zsize*in_gbox->zsize) / 2;
    }else {
	int seq = (cellID - offset_for_level(level)) & 0x07;  // last 3 bits 
	int parentID = parent_of(cellID);
	
	int fact = 1 << (in_max_level - level);
	
	iris_real dx = fact * in_leaf_size[0];
	iris_real dy = fact * in_leaf_size[1];
	iris_real dz = fact * in_leaf_size[2];
	
	if(seq & 0x01) {
	    geomc[0] = in_meta[parentID].geomc[0] + dx / 2;
	}else {
	    geomc[0] = in_meta[parentID].geomc[0] - dx / 2;
	}
	if(seq & 0x02) {
	    geomc[1] = in_meta[parentID].geomc[1] + dy / 2;
	}else {
	    geomc[1] = in_meta[parentID].geomc[1] - dy / 2;
	}
	if(seq & 0x04) {
	    geomc[2] = in_meta[parentID].geomc[2] + dz / 2;
	}else {
	    geomc[2] = in_meta[parentID].geomc[2] - dz / 2;
	}
	
	maxr = sqrt(dx*dx + dy*dy + dz*dz) / 2;
    }

    // rank = -1;
    // if(level >= in_local_root_level) {
    // 	int t_id = cellID;
    // 	int t_level = level;
    // 	while(t_level > in_local_root_level) {
    // 	    t_id = parent_of(t_id);
    // 	    t_level = level_of(t_id);
    // 	}
    // 	int idx = t_id - offset_for_level(t_level);
    // 	int bits = int(log(in_comm_size)/M_LN2);
    // 	int shift = (3 - bits % 3) % 3;
    // 	rank = idx >> shift;
    // }
}

void cell_t::compute_ses(particle_t *in_particles)
{
    // TODO: maybe do this without memory alloc (e.g. use m_ncrit as maximum and put assert that it's enough)
    point_t *points = (point_t *)memory::wmalloc(num_children * sizeof(point_t));
    for(int i=0;i<num_children;i++) {
	points[i].r[0] = in_particles[first_child+i].xyzq[0];
	points[i].r[1] = in_particles[first_child+i].xyzq[1];
	points[i].r[2] = in_particles[first_child+i].xyzq[2];
    }
    ses_of_points(points, num_children, &ses);
    memory::wfree(points);
}

IRIS_CUDA_DEVICE_HOST int cell_meta_t::offset_for_level(int level)
{
    return ((1 << 3 * level)-1) / 7;
}

//
// Given a cellID, determine the level of the cell
//
IRIS_CUDA_DEVICE_HOST int cell_meta_t::level_of(int in_cellID)
{
    int retval = -1;
    for(int i=in_cellID;i>=0;i-=(1 << 3 * retval)) {
	retval++;
    }
    return retval;
}

//
// Given a cellID, find the cellID of its parent
//
IRIS_CUDA_DEVICE_HOST int cell_meta_t::parent_of(int in_cellID)
{
    int level = level_of(in_cellID);
    int curr_off = offset_for_level(level);
    int parent_off = offset_for_level(level-1);
    int retval = ((in_cellID - curr_off) >> 3) + parent_off;
    return retval;
}


IRIS_CUDA_DEVICE_HOST int cell_meta_t::leaf_coords_to_ID(int lx, int ly, int lz, int max_level)
{
    int offset = cell_meta_t::offset_for_level(max_level);
    int nd = 1 << max_level;
    
    int lc[] = { lx % nd, ly % nd, lz % nd };

    int id = 0;
    for(int l=0;l<max_level; l++) {
	for(int d=0;d<3;d++) {
	    id += (lc[d] & 1) << (3*l + d);
	    lc[d] >>= 1;
	}
    }
    return offset + id;
}
