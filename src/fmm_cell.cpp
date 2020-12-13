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
#include "fmm_cell.h"

using namespace ORG_NCSA_IRIS;

//
// Determine the center of the cell by using its cellID, the global box and the leaf size
//
void cell_meta_t::set(cell_meta_t *in_meta, int cellID, const box_t<iris_real> *in_gbox, iris_real *in_leaf_size, int in_max_level,
		      int in_comm_size, int in_local_root_level)
{
    if(cellID == 0) {
	center[0] = (in_gbox->xlo + in_gbox->xhi)/2;
	center[1] = (in_gbox->ylo + in_gbox->yhi)/2;
	center[2] = (in_gbox->zlo + in_gbox->zhi)/2;
	radius = sqrt(in_gbox->xsize*in_gbox->xsize + in_gbox->ysize*in_gbox->ysize + in_gbox->zsize*in_gbox->zsize) / 2;
	return;
    }

    int level = level_of(cellID);
    int seq = (cellID - offset_for_level(level)) & 0x07;  // last 3 bits 
    int parentID = parent_of(cellID);
    
    int fact = 1 << (in_max_level - level);

    iris_real dx = fact * in_leaf_size[0];
    iris_real dy = fact * in_leaf_size[1];
    iris_real dz = fact * in_leaf_size[2];

    if(seq & 0x01) {
	center[0] = in_meta[parentID].center[0] + dx / 2;
    }else {
	center[0] = in_meta[parentID].center[0] - dx / 2;
    }
    if(seq & 0x02) {
	center[1] = in_meta[parentID].center[1] + dy / 2;
    }else {
	center[1] = in_meta[parentID].center[1] - dy / 2;
    }
    if(seq & 0x04) {
	center[2] = in_meta[parentID].center[2] + dz / 2;
    }else {
	center[2] = in_meta[parentID].center[2] - dz / 2;
    }
    
    radius = sqrt(dx*dx + dy*dy + dz*dz) / 2;

    rank = -1;
    if(level >= in_local_root_level) {
	int t_id = cellID;
	int t_level = level;
	while(t_level > in_local_root_level) {
	    t_id = parent_of(t_id);
	    t_level = level_of(t_id);
	}
	int idx = t_id - offset_for_level(t_level);
	int bits = int(log(in_comm_size)/M_LN2);
	int shift = (3 - bits % 3) % 3;
	rank = idx >> shift;
    }
}


//
// A comparator function used to sort the array of xparticles by cellID ascending
//
// static int __xcompar_cells_asc(const void *aptr, const void *bptr)
// {
//     cell_t *a = (cell_t *)aptr;
//     cell_t *b = (cell_t *)bptr;
    
//     if(a->cellID > b->cellID) {
// 	return 1;
//     }else if(a->cellID < b->cellID) {
// 	return -1;
//     }else {
// 	return 0;
//     }
// }

// //
// // A comparator function used to sort the array of particles by cellID descending
// //
// static int __xcompar_cells_desc(const void *aptr, const void *bptr)
// {
//     cell_t *a = (cell_t *)aptr;
//     cell_t *b = (cell_t *)bptr;
    
//     if(a->cellID < b->cellID) {
// 	return 1;
//     }else if(a->cellID > b->cellID) {
// 	return -1;
//     }else {
// 	return 0;
//     }
// }

// void cell_t::sort(cell_t *in_out_data, int count, bool desc)
// {
//     int (*fn)(const void *, const void *);
//     if(desc) {
// 	fn = __xcompar_cells_desc;
//     }else {
// 	fn = __xcompar_cells_asc;
//     }
//     qsort(in_out_data, count, sizeof(cell_t), fn);
// }
