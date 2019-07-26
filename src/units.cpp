// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2019, the National Center for Supercomputing Applications
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
#include "units.h"

using namespace ORG_NCSA_IRIS;

units::units(EUnits style)
{
    switch(style) {
    case real:
	init_real();
	break;

    case md:
	init_md();
	break;

    default:
	throw std::invalid_argument("Unit style not supported!");
    }

}

units::~units()
{
}

// Length: Angstrom
// Charge: # of Elementary charges
// Energy: Kcal/mol
void units::init_real()
{
    this->e  = 1.0;
    this->ecf = 332.0637129954289;
    this->ang = 1.0;
    this->energy_unit = "Kcal/mol";
    this->length_unit = "ang";
}


// Length: nm
// Charge: # of Elementary charges
// Energy: KJ/mol
void units::init_md()
{
    this->e  = 1.0;
    this->ecf = 138.93545751728743;
    this->ang = 0.1;
    this->energy_unit = "KJ/mol";
    this->length_unit = "nm";
}

