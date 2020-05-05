// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2018, the National Center for Supercomputing Applications
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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_PROC_GRID_H__
#define __IRIS_PROC_GRID_H__

#include "grid.h"

namespace ORG_NCSA_IRIS {

#define IRIS_LAYOUT_UNDEFINED -1
    
    /////////////////////////////////////////////////////////////
    /// Plane domain decomposition - faster
    ///
    /// It turns out that the fastest possible way to split work
    /// between P3M nodes is make so that a single processor
    /// "owns" one or more 2D planes. This is possible only if
    /// the number of processors is less than or equal to the
    /// number of mesh nodes in one of the directions.
    /// There are 3 possible scenarios:
    ///   - own a set of YZ planes along the X direction
    ///   - own a set of XY planes along the Z direction
    ///   - own a set of ZX planes along the Y direction
    /////////////////////////////////////////////////////////////

#define IRIS_LAYOUT_PLANES_YZ  0  // YZ planes in the X direction
#define IRIS_LAYOUT_PLANES_XY  1  // XY planes in the Z direction
#define IRIS_LAYOUT_PLANES_ZX  2  // ZX planes in the Y direction

    
    /////////////////////////////////////////////////////////////
    /// Pencil domain decomposition - slower
    ///
    /// If the number of processors is bigger than the number of
    /// mesh nodes in one direction (e.g. 256 processors for
    /// 128x128x128 mesh), then we need to resort to pencils --
    /// a processor owns a set of "pencils" which span a whole
    /// dimension, for example the 128x128x128 mesh is split into
    /// 256 8x8x128 local meshes.
    /// There are 3 possible scenarios:
    ///   - own a set of Z pencils along the XY plane
    ///   - own a set of Y pencils along the ZX plane
    ///   - own a set of X pencils along the YZ plane
    /////////////////////////////////////////////////////////////
    
#define IRIS_LAYOUT_PENCILS_Z  3  // pencils along Z in the XY plane
#define IRIS_LAYOUT_PENCILS_Y  4  // pencils along Z in the XY plane
#define IRIS_LAYOUT_PENCILS_X  5  // pencils along Z in the XY plane

    class proc_grid : public grid {

    public:
	proc_grid(class iris *obj);
	~proc_grid();

	void commit();

	int get_layout() { return m_layout; };

    private:
	void figure_out_layout();
	
    private:
	int m_layout;
    };
}

#endif
