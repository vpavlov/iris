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

#include <stdexcept>
#include "state_accessor.h"
#include "box.h"

namespace ORG_NCSA_IRIS {

    class domain : protected state_accessor {

    public:
	domain(class iris *obj);
	~domain();

	void set_global_box(iris_real x0, iris_real y0, iris_real z0,
			    iris_real x1, iris_real y1, iris_real z1);

	void set_periodicity(int dir, int in_value)
	{
	    if(dir < 0 || dir > 2) {
		throw std::invalid_argument("Invalid direction for periodicity!");
	    }

	    if(in_value < 0 || in_value > 1) {
		throw std::invalid_argument("Invalid periodicity!");
	    }

	    m_pbc[dir] = in_value;
	}

	void set_periodocity(int in_x, int in_y, int in_z)
	{
	    set_periodicity(0, in_x);
	    set_periodicity(1, in_y);
	    set_periodicity(2, in_z);
	}

	void commit();

    public:
	bool m_initialized;
	int m_pbc[3];   // periodicity of the box for each direction
	box_t<iris_real> m_global_box;
	box_t<iris_real> m_local_box;

    private:
	bool m_dirty;
    };
}
#endif
