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
#include <stdexcept>
#include <vector>
#include <tuple>
#include "iris.h"
#include "charge_assigner.h"
#include "mesh.h"
#include "proc_grid.h"
#include "comm_rec.h"

#include "memory.h"
#include "domain.h"
#include "logger.h"
#include "event.h"

using namespace ORG_NCSA_IRIS;

static iris_real coeff2[2][2] =
    {{ 1/2.0,  1.0 },
     { 1/2.0, -1.0 }};

static iris_real coeff3[3][3] =
    {{ 1/8.0,  1/2.0,  1/2.0 },
     { 6/8.0,      0, -2/2.0 },
     { 1/8.0, -1/2.0,  1/2.0 }};

static iris_real coeff4[4][4] = 
    {{  1/48.0,  1/8.0,  1/4.0,  1/6.0 },
     { 23/48.0,  5/8.0, -1/4.0, -3/6.0 },
     { 23/48.0, -5/8.0, -1/4.0,  3/6.0 },
     {  1/48.0, -1/8.0,  1/4.0, -1/6.0 }};

static iris_real coeff5[5][5] =
    {{   1/384.0,   1/48.0,   1/16.0,  1/12.0,  1/24.0 },
     {  76/384.0,  22/48.0,   4/16.0, -2/12.0, -4/24.0 },
     { 230/384.0,        0, -10/16.0,       0,  6/24.0 },
     {  76/384.0, -22/48.0,   4/16.0,  2/12.0, -4/24.0 },
     {   1/384.0,  -1/48.0,   1/16.0, -1/12.0,  1/24.0 }};

static iris_real coeff6[6][6] =
    {{    1/3840.0,    1/384.0,   1/96.0,   1/48.0,  1/48.0,   1/120.0 },
     {  237/3840.0,   75/384.0,  21/96.0,   3/48.0, -3/48.0,  -5/120.0 },
     { 1682/3840.0,  154/384.0, -22/96.0, -14/48.0,  2/48.0,  10/120.0 },
     { 1682/3840.0, -154/384.0, -22/96.0,  14/48.0,  2/48.0, -10/120.0 },
     {  237/3840.0,  -75/384.0,  21/96.0,  -3/48.0, -3/48.0,   5/120.0 },
     {    1/3840.0,   -1/384.0,   1/96.0,  -1/48.0,  1/48.0,  -1/120.0 }};

static iris_real coeff7[7][7] =
    {{     1/46080.0,     1/3840.0,    1/768.0,   1/288.0,   1/192.0,  1/240.0,   1/720.0 },
     {   722/46080.0,   236/3840.0,   74/768.0,  20/288.0,   2/192.0, -4/240.0,  -6/720.0 },
     { 10543/46080.0,  1445/3840.0,   79/768.0, -43/288.0, -17/192.0,  5/240.0,  15/720.0 },
     { 23548/46080.0,            0, -308/768.0,         0,  28/192.0,        0, -20/720.0 },
     { 10543/46080.0, -1445/3840.0,   79/768.0,  43/288.0, -17/192.0, -5/240.0,  15/720.0 },
     {   722/46080.0,  -236/3840.0,   74/768.0, -20/288.0,   2/192.0,  4/240.0,  -6/720.0 },
     {     1/46080.0,    -1/3840.0,    1/768.0,  -1/288.0,   1/192.0, -1/240.0,   1/720.0 }};

charge_assigner::charge_assigner(iris *obj)
    :state_accessor(obj), m_order(0), m_dirty(true), m_weights(NULL)
{
}

charge_assigner::~charge_assigner()
{
    memory::destroy_2d(m_weights);
}

void charge_assigner::set_order(int in_order)
{
    if(in_order < 2 || in_order > 7) {
	throw std::invalid_argument("Orders below 2 and above 7 are not supported!");
    }

    m_order = in_order;
    m_dirty = true;
    m_logger->trace("Charge assignment/interpolation order set to %d", m_order);
}

void charge_assigner::commit()
{
    // set default values
    if(m_order == 0) {
	set_order(2);
    }

    if(m_dirty) {
	m_ics_from = -(m_order-1)/2;
	m_ics_to = m_order/2;
	
	if(m_order % 2) {
	    m_ics_bump = 0.5;
	    m_ics_center = 0.0;
	}else {
	    m_ics_bump = 0.0;
	    m_ics_center = 0.5;
	}
	
	if(m_order == 2) {
	    m_coeff = (iris_real *) coeff2;
	}else if(m_order == 3) {
	    m_coeff = (iris_real *) coeff3;
	}else if(m_order == 4) {
	    m_coeff = (iris_real *) coeff4;
	}else if(m_order == 5) {
	    m_coeff = (iris_real *) coeff5;
	}else if(m_order == 6) {
	    m_coeff = (iris_real *) coeff6;
	}else if(m_order == 7) {
	    m_coeff = (iris_real *) coeff7;
	}

	if(m_weights != NULL) {
	    memory::destroy_2d(m_weights);
	}
	memory::create_2d(m_weights, 3, m_order);
	m_dirty = false;
    }
}


void charge_assigner::compute_weights(iris_real dx, iris_real dy, iris_real dz)
{
    iris_real r1, r2, r3;

    for(int i = 0; i < m_order; i++) {
	r1 = r2 = r3 = (iris_real)0.0;
	
	for(int j = m_order - 1; j >= 0; j--) {
	    r1 = m_coeff[i*m_order + j] + r1 * dx;
	    r2 = m_coeff[i*m_order + j] + r2 * dy;
	    r3 = m_coeff[i*m_order + j] + r3 * dz;
	}
	
	m_weights[0][i] = r1;
	m_weights[1][i] = r2;
	m_weights[2][i] = r3;
    }
}
