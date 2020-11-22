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
#include "fmm_cell.h"

using namespace ORG_NCSA_IRIS;

//
// Determine the center of the cell by using its cellID, the global box and the leaf size
//
void cell_t::set_center(const box_t<iris_real> *in_gbox, iris_real *in_leaf_size)
{
    int level = level_of(cellID);
    int index = cellID - offset_for_level(level);
    int nd = 1 << level;
    int iz = index % nd;
    index /= nd;
    int iy = index % nd;
    int ix = index / nd;

    center[0] = (ix + 0.5) * in_leaf_size[0] + in_gbox->xlo;
    center[1] = (iy + 0.5) * in_leaf_size[1] + in_gbox->ylo;
    center[2] = (iz + 0.5) * in_leaf_size[2] + in_gbox->zlo;
}
