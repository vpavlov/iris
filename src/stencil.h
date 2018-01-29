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
#ifndef __IRIS_STENCIL_H__
#define __IRIS_STENCIL_H__

namespace ORG_NCSA_IRIS {
    //
    // Representation of a stencil used to approximate a derivative operator.
    // 
    // A stencil is a geometric arrangement of a nodal group that relate to the
    // point of interest by using a numerical approximation routine.
    // 
    // The coefficients of the stencil in the different nodes are stored in the
    // array, m_data. The class doesn't make any assumptions on the
    // dimensionality m_dim of the stencil, it can be 1D, 2D, etc. This is left
    // to the discretion of the derived classes. The accuracy order of the
    // stencil (m_order) is a characteristic which the derived classes may
    // use to calculate the amount of data necessary. 
    //
    class stencil {

    public:
	stencil(int in_dim, int in_order, int in_acc);
	~stencil();

	virtual void commit() = 0;
	virtual void trace(const char *in_name) = 0;

    public:
	bool m_dirty;  // wether to recalculate on commit
	int m_dim;     // dimensionality (e.g. 1D, 2D, 3D)
	int m_order;   // first, second, third derivative
	int m_acc;     // accuracy order in h
	void *m_data;  // coefficients of the stencil
    };
}

#endif
