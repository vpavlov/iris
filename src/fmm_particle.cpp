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
#include <stdlib.h>
#include "fmm_particle.h"

using namespace ORG_NCSA_IRIS;

//
// A comparator function used to sort the array of xparticles by cellID ascending
//
int ORG_NCSA_IRIS::__xcompar_asc(const void *aptr, const void *bptr)
{
    particle_t *a = (particle_t *)aptr;
    particle_t *b = (particle_t *)bptr;
    
    if(a->cellID > b->cellID) {
	return 1;
    }else if(a->cellID < b->cellID) {
	return -1;
    }else {
	return 0;
    }
}

//
// A comparator function used to sort the array of particles by cellID descending
//
int ORG_NCSA_IRIS::__xcompar_desc(const void *aptr, const void *bptr)
{
    particle_t *a = (particle_t *)aptr;
    particle_t *b = (particle_t *)bptr;
    
    if(a->cellID < b->cellID) {
	return 1;
    }else if(a->cellID > b->cellID) {
	return -1;
    }else {
	return 0;
    }
}

//
// A comparator function used to sort the array of particles by rank and then by index
//
int ORG_NCSA_IRIS::__compar_id_asc(const void *aptr, const void *bptr)
{
    particle_t *a = (particle_t *)aptr;
    particle_t *b = (particle_t *)bptr;

    if(a->rank > b->rank) {
	return 1;
    }else if(a->rank < b->rank) {
	return -1;
    }else {
	if(a->index > b->index) {
	    return 1;
	}else if(a->index < b->index) {
	    return -1;
	}else {
	    return 0;
	}
    }
}

void ORG_NCSA_IRIS::sort_back_particles(particle_t *in_out_particles, int count)
{
    int (*fn)(const void *, const void *);
    fn = __compar_id_asc;
    qsort(in_out_particles, count, sizeof(particle_t), fn);
}
