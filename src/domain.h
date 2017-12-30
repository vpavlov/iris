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
	void set_box(iris_real in_box_min[3], iris_real in_box_max[3]);

	void setup_local_box();

    public:
	int  dimensions;  // # of dimensions of the domain, default 3

	// global box
	iris_real box_min[3];  // left/bottom/front corner of computational box
	iris_real box_max[3];  // right/top/back corner of the computational box
	iris_real box_size[3]; // min - max in every dimention

	// local box (sub-domain for which this proc is responsicle)
	iris_real loc_box_min[3];
	iris_real loc_box_max[3];
	iris_real loc_box_size[3];
    };
}
#endif
