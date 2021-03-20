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
#ifndef __IRIS_BOX_H__
#define __IRIS_BOX_H__

#include <cmath>

namespace ORG_NCSA_IRIS {

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

    template <typename T>
    struct box_t {
	T xlo, ylo, zlo;
	T xhi, yhi, zhi;
	T xsize, ysize, zsize;

	box_t(int dummy = 0) {}; // to satisfy the compiler for create_1d
	
	bool in(iris_real *charge)
	{
	    iris_real x = charge[0];
	    iris_real y = charge[1];
	    iris_real z = charge[2];
	    
	    return ((x >= xlo && x < xhi) &&
		    (y >= ylo && y < yhi) &&
		    (z >= zlo && z < zhi));
	}

	// bool in(iris_real *charge)
	// {
	//     iris_real x = charge[0];
	//     iris_real y = charge[1];
	//     iris_real z = charge[2];
	    
	//     return ((!(x < xlo) && x < xhi) &&
	// 	    (!(y < ylo) && y < yhi) &&
	// 	    (!(z < zlo) && z < zhi));
	// }

	box_t<T> operator && (box_t<T> &other)
	{
	    box_t<T> retval;
	    retval.xlo = MAX(this->xlo, other.xlo);
	    retval.xhi = MIN(this->xhi, other.xhi);

	    retval.ylo = MAX(this->ylo, other.ylo);
	    retval.yhi = MIN(this->yhi, other.yhi);

	    retval.zlo = MAX(this->zlo, other.zlo);
	    retval.zhi = MIN(this->zhi, other.zhi);

	    if(retval.xlo > retval.xhi ||
	       retval.ylo > retval.yhi ||
	       retval.zlo > retval.zhi)
	    {
		retval.xlo = retval.ylo = retval.zlo = 
		    retval.xhi = retval.yhi = retval.zhi = 
		    retval.xsize = retval.ysize = retval.zsize = 0;
	    }else {
		retval.xsize = retval.xhi - retval.xlo + 1;
		retval.ysize = retval.yhi - retval.ylo + 1;
		retval.zsize = retval.zhi - retval.zlo + 1;
	    }
	    return retval;
	}

	iris_real distance_to(iris_real x, iris_real y, iris_real z)
	{
	    iris_real dx = (x > xhi) * (x - xhi) + (x < xlo) * (xlo - x);
	    iris_real dy = (y > yhi) * (y - yhi) + (y < ylo) * (ylo - y);
	    iris_real dz = (z > zhi) * (z - zhi) + (z < zlo) * (zlo - z);
	    return sqrt(dx*dx + dy*dy + dz*dz);
	}

	bool periodic_overlap(box_t<iris_real> *other_box, box_t<iris_real> *gbox)
	{
	    if(xlo < gbox->xlo) {
		box_t<iris_real> b1;
		b1.xlo = gbox->xsize + xlo;
		b1.xhi = gbox->xhi;
		b1.ylo = ylo;
		b1.yhi = yhi;
		b1.zlo = zlo;
		b1.zhi = zhi;
	    
		box_t<iris_real> b2;
		b2.xlo = gbox->xlo;
		b2.xhi = xhi;
		b2.ylo = ylo;
		b2.yhi = yhi;
		b2.zlo = zlo;
		b2.zhi = zhi;
	    
		return b1.periodic_overlap(other_box, gbox) || b2.periodic_overlap(other_box, gbox);
	    }

	    if(xhi > gbox->xhi) {
		box_t<iris_real> b1;
		b1.xlo = xlo;
		b1.xhi = gbox->xhi;
		b1.ylo = ylo;
		b1.yhi = yhi;
		b1.zlo = zlo;
		b1.zhi = zhi;
	    
		box_t<iris_real> b2;
		b2.xlo = gbox->xlo;
		b2.xhi = xhi - gbox->xsize;
		b2.ylo = ylo;
		b2.yhi = yhi;
		b2.zlo = zlo;
		b2.zhi = zhi;
	    
		return b1.periodic_overlap(other_box, gbox) || b2.periodic_overlap(other_box, gbox);
	    }

	    if(ylo < gbox->ylo) {
		box_t<iris_real> b1;
		b1.xlo = xlo;
		b1.xhi = xhi;
		b1.ylo = gbox->ysize + ylo;
		b1.yhi = gbox->yhi;
		b1.zlo = zlo;
		b1.zhi = zhi;
	    
		box_t<iris_real> b2;
		b2.xlo = xlo;
		b2.xhi = xhi;
		b2.ylo = gbox->ylo;
		b2.yhi = yhi;
		b2.zlo = zlo;
		b2.zhi = zhi;
	    
		return b1.periodic_overlap(other_box, gbox) || b2.periodic_overlap(other_box, gbox);
	    }

	    if(yhi > gbox->yhi) {
		box_t<iris_real> b1;
		b1.xlo = xlo;
		b1.xhi = xhi;
		b1.ylo = ylo;
		b1.yhi = gbox->yhi;
		b1.zlo = zlo;
		b1.zhi = zhi;
	    
		box_t<iris_real> b2;
		b2.xlo = xlo;
		b2.xhi = xhi;
		b2.ylo = gbox->ylo;
		b2.yhi = yhi - gbox->ysize;
		b2.zlo = zlo;
		b2.zhi = zhi;
	    
		return b1.periodic_overlap(other_box, gbox) || b2.periodic_overlap(other_box, gbox);
	    }


	    if(zlo < gbox->zlo) {
		box_t<iris_real> b1;
		b1.xlo = xlo;
		b1.xhi = xhi;
		b1.ylo = ylo;
		b1.yhi = yhi;
		b1.zlo = gbox->zsize + zlo;
		b1.zhi = gbox->zhi;
	    
		box_t<iris_real> b2;
		b2.xlo = xlo;
		b2.xhi = xhi;
		b2.ylo = ylo;
		b2.yhi = yhi;
		b2.zlo = gbox->zlo;
		b2.zhi = zhi;
	    
		return b1.periodic_overlap(other_box, gbox) || b2.periodic_overlap(other_box, gbox);
	    }

	    if(zhi > gbox->zhi) {
		box_t<iris_real> b1;
		b1.xlo = xlo;
		b1.xhi = xhi;
		b1.ylo = ylo;
		b1.yhi = yhi;
		b1.zlo = zlo;
		b1.zhi = gbox->zhi;
	    
		box_t<iris_real> b2;
		b2.xlo = xlo;
		b2.xhi = xhi;
		b2.ylo = ylo;
		b2.yhi = yhi;	
		b2.zlo = gbox->zlo;
		b2.zhi = zhi - gbox->zsize;
    
		return b1.periodic_overlap(other_box, gbox) || b2.periodic_overlap(other_box, gbox);
	    }

	    box_t<iris_real> tmp = (*this) && *other_box;
	    if(tmp.xsize > 1 && tmp.ysize > 1 && tmp.zsize > 1) {
		return true;
	    }
	
	    return false;
	}
	
    };

}

#endif
