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
#ifndef __IRIS_FMM_PARTICLE_H__
#define __IRIS_FMM_PARTICLE_H__

#include "real.h"

namespace ORG_NCSA_IRIS {

    // When constructing the FMM tree, we start with atoms. Each processor has been
    // assigned a spatial area of the global domain and sent information about all
    // the atoms that reside in its local box. These atoms potentially may come from
    // different client processors.
    //
    // In the FMM tree, the particles are the lowest form of life; they live in the
    // dungeons -- one level below the lowest level of cells. That's why they need
    // a special structure. In this structure we keep the reference to the original
    // atom (rank of the sender and its # in that sender's array) and a reference
    // to the cell (leaf) in which this particle resides.
    
    struct particle_t {
	iris_real xyzq[4];
	int cellID;
	int rank;           // from which rank this particle came ?
	int index;          // # in m_charges{rank}
	iris_real tgt[4];   // φ, Ex, Ey, Ez
	int dummy[5];
    };

    int __compar_id_asc(const void *aptr, const void *bptr);
    

    void sort_back_particles(particle_t *in_out_particles, int count);
}

#endif
