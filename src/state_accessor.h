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
#ifndef __IRIS_IRIS_STATE_ACCESSOR_H__
#define __IRIS_IRIS_STATE_ACCESSOR_H__

#include "iris.h"

namespace ORG_NCSA_IRIS {

    // This is a neat trick lifted from LAMMPS:
    // 
    // As we all know, encapsulation in OOP doesn't work. We need access to
    // the state. The state is packaged inside various objects to make
    // comprehension possible. However, in a complex system like this,
    // virtually every such object needs access to some part of the state that
    // is packaged in another object. If we go by the book, we should do
    // something like this:
    //   - pass "parent" IRIS object to object x;
    //   - inside object x do m_iris->get_y()->do_something();
    // Instead, we just make y public in IRIS and do this:
    //   - pass "parent" IRIS object to object x;
    //   - y->get_something();
    // For this to work, object x must have a member y; this is accomplished
    // by this helper class that all x's inherit and which receives upon
    // construction a pointer to the "parent" IRIS object and then through
    // C++ reference machinery adds references inside x to pointers in the 
    // parent object.

    class state_accessor {

    public:
	state_accessor(iris *obj) :
	    m_iris(obj),
	    m_queue(obj->m_queue),
	    m_uber_comm(obj->m_uber_comm),
	    m_local_comm(obj->m_local_comm),
	    m_inter_comm(obj->m_inter_comm),
	    m_logger(obj->m_logger),

	    m_domain(obj->m_domain),
	    m_proc_grid(obj->m_proc_grid),
	    m_mesh(obj->m_mesh),
	    m_chass(obj->m_chass)
	{};

	virtual ~state_accessor() {};

    protected:
	iris *m_iris;
	event_queue *&m_queue;
	comm_rec *&m_uber_comm;
	comm_rec *&m_local_comm;
	comm_rec *&m_inter_comm;
	logger *&m_logger;
	domain *&m_domain;
	proc_grid *&m_proc_grid;
	mesh *&m_mesh;
	charge_assigner *&m_chass;
    };

}

#endif
