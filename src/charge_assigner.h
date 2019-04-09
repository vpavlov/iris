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
#ifndef __IRIS_CHARGE_ASSIGNER_H__
#define __IRIS_CHARGE_ASSIGNER_H__

#include <tuple>
#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

    class charge_assigner : protected state_accessor {

    public:
	charge_assigner(class iris *in_obj);
	~charge_assigner();

	void set_order(int in_order);
	void commit();

	void compute_weights(iris_real dx, iris_real dy, iris_real dz);

    private:

	bool m_dirty;  // dirty configuration, must re-calculate on commit

	//
	// Charge assignment/interpolation machinery
	// 
	// m_order: the order of the interpolation. Example: if order = 2, then
	// a charge is distributed to the nearest 2 points of the charge_assigner in each
	// direction (total of 2^3 = 8 points). If order = 3, then a charge is
	// distributed to the nearest 3 points of the charge_assigner in each direction
	// (total of 3^3 = 27 points). Orders between 2 and 7 are supported.
    public:
	int m_order;         // charge assignment/interpolation order
	iris_real *m_gfd_coeff;  // Hockney/Eastwood modified Green function denominator coefficients


	// Finding the nearest N points to a charge (where N = order)
	// 
	// Local interpolation coordinate system (ics):
	// We have a coordinate system where the node immediately
	// "left" of a charge has coordinate 0.0 and the node immediately
	// "right" of the charge has coordinate 1.0
	//
	// Even orders:
	// ------------
	// For order 2: distributed between         0, 1
	// For order 4: distributed between     -1, 0, 1, 2
	// For order 6: distributed between -2, -1, 0, 1, 2, 3
	// 
	// Odd orders:
	// -----------
	// It would seem that for order 3 we would need -1, 0 and 1. However,
	// there is a twist: if a charge is at e.g. 0.9, then the 3 nearest
	// points are 0, 1 and 2 instead: 2.0 - 0.9 = 1.1, while 0.9 - -1 = 1.9
	// Thus, for charges that at coord > 0.5, we need to move the "left"
	// point to the right.
	//
	// This can be accomplished by adding 0.5 to the coordinates before
	// truncating, for odd orders. For even orders we add 0
	//
	// So we have:
	//
	// m_ics_from: the leftmost node (in ics coords) that participates
	// m_ics_to: the rightmost node that participates in the interpolation
	// m_ics_center: the center of the ics system (0.5 for even, 0 for odd)
	// m_ics_bump: correction to bump to nearest (0.5 for odd, 0 for even)
	//
	// It is easy to see that the following relations hold
	int       m_ics_from;    // -(order-1)/2
	int       m_ics_to;      // order/2
	iris_real m_ics_center;  // even orders: 0.5; odd order: 0.0
	iris_real m_ics_bump;    // even orders: 0.0; odd orders: 0.5

	iris_real ***m_weights;  // interpolation weights, depend on charge pos

    private:
	iris_real *m_coeff;      // interpolation coefficients, depend on order

    };
}
#endif
