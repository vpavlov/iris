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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_DOMAIN_H__
#define __IRIS_DOMAIN_H__

#include "global_state.h"

namespace ORG_NCSA_IRIS {

    class domain : protected global_state {

    public:
	domain(class iris *obj);
	~domain();

	void set_dimensions(int in_dim);
	void set_box(iris_real x0, iris_real y0, iris_real z0,
		     iris_real x1, iris_real y1, iris_real z1);
	void setup_local();

    public:
	int  dimensions;  // # of dimensions of the domain, default 3

	// Global and local cartesian boxes are described by the 6 coordinates
	// of each of the sides (which are parallel to the coordinate axis)
	//
	// box_sides[0][0] - x coord of left side
	// box_sides[0][1] - y coord of bottom side
	// box_sides[0][2] - z coord of front side
	// box_sides[1][0] - x coord of right side
	// box_sides[1][1] - y coord of top side
	// box_sides[1][2] - z coord of back side
	//
	// Another way to put it:
	// box_sides[0] are the coordinates of the left/bottom/front corner;
	// box_sides[1] are the coordinates of the right/top/back corner;
	//
	iris_real box_sides[2][3];
	iris_real box_size[3];  // size in every direction

	iris_real lbox_sides[2][3];
	iris_real lbox_size[3];
    };
}
#endif
