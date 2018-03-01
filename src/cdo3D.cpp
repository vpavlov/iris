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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include <stdio.h>
#include "cdo3D.h"
#include "memory.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

cdo3D::cdo3D(int n, iris_real c, int xp, int yp, int zp)
{
    m_n = n;
    memory::create_3d(m_data, n+1, n+1, n+1, true);
    int center = n / 2;
    int *x_cdo = coeff(xp);
    int *y_cdo = coeff(yp);
    int *z_cdo = coeff(zp);
    int sx = center - xp/2;
    int ex = sx + xp;
    for(int x = sx; x <= ex; x++ ) {
	iris_real v1 = c * x_cdo[x-sx];
	int sy = center - yp/2;
	int ey = sy + yp;
	for(int y = sy; y <= ey; y++) {
	    iris_real v2 = v1 * y_cdo[y-sy];
	    int sz = center - zp/2;
	    int ez = sz + zp;
	    for(int z = sz; z <= ez; z++) {
		m_data[x][y][z] = v2 * z_cdo[z-sz];
	    }
	}
    }
    memory::destroy_1d(x_cdo);
    memory::destroy_1d(y_cdo);
    memory::destroy_1d(z_cdo);
}

cdo3D::~cdo3D()
{
    memory::destroy_3d(m_data);
}

int *cdo3D::coeff(int n)
{
    int *c;
    memory::create_1d(c, n+1);
    int sign = 1;
    for(int k=0;k<=n;k++) {
	c[k] = sign * binom(n, k);
	sign *= -1;
    }
    return c;
}

void cdo3D::dump()
{
    for(int i=0;i<=m_n;i++) {
	for(int j=0;j<=m_n;j++) {
	    for(int k=0;k<=m_n;k++) {
		printf("cdo[%d][%d][%d] = %g\n", i, j, k, m_data[i][j][k]);
	    }
	    printf("\n");
	}
    }
}

void cdo3D::operator += (cdo3D &other)
{
    for(int i=0;i<=m_n;i++) {
	for(int j=0;j<=m_n;j++) {
	    for(int k=0;k<=m_n;k++) {
		m_data[i][j][k] += other.m_data[i][j][k];
	    }
	}
    }
}
