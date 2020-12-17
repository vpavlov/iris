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
#ifndef __IRIS_POINT_H__
#define __IRIS_POINT_H__

namespace ORG_NCSA_IRIS {

    struct point_t {
	iris_real r[3];

	inline iris_real dot(point_t *b)
	{
	    return r[0]*b->r[0] + r[1]*b->r[1] + r[2]*b->r[2];
	}
	
	inline point_t cross(point_t *b)
	{
	    point_t rr;
	    rr.r[0] = this->r[1] * b->r[2] - this->r[2] * b->r[1];
	    rr.r[1] = this->r[2] * b->r[0] - this->r[0] * b->r[2];
	    rr.r[2] = this->r[0] * b->r[1] - this->r[1] * b->r[0];
	    return rr;
	}

	inline point_t minus(point_t *b)
	{
	    point_t rr;
	    rr.r[0] = this->r[0] - b->r[0];
	    rr.r[1] = this->r[1] - b->r[1];
	    rr.r[2] = this->r[2] - b->r[2];
	    return rr;
	}

	inline point_t plus(point_t *b)
	{
	    point_t rr;
	    rr.r[0] = this->r[0] + b->r[0];
	    rr.r[1] = this->r[1] + b->r[1];
	    rr.r[2] = this->r[2] + b->r[2];
	    return rr;
	}
	
    };
}
    
#endif
