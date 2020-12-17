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
#include <math.h>
#include <stdexcept>
#include "ses.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

static void sphere_through(point_t *R, int nr, sphere_t *out_ses)
{
    if(nr == 0) {
	out_ses->c.r[0] = 0.0;
	out_ses->c.r[1] = 0.0;
	out_ses->c.r[2] = 0.0;
	out_ses->r = 0.0;
    }else if(nr == 1) {
	out_ses->c.r[0] = R[0].r[0];
	out_ses->c.r[1] = R[0].r[1];
	out_ses->c.r[2] = R[0].r[2];
	out_ses->r = 0.0;
    }else if(nr == 2) {
	point_t c = R[0].plus(R+1);
	out_ses->c.r[0] = c.r[0] * (iris_real)0.5;
	out_ses->c.r[1] = c.r[1] * (iris_real)0.5;
	out_ses->c.r[2] = c.r[2] * (iris_real)0.5;
	
	point_t r = out_ses->c.minus(R+0);
	out_ses->r = sqrt(r.dot(&r));
    }else if(nr == 3) {
	point_t a = R[0].minus(R+2);
	point_t b = R[1].minus(R+2);

	iris_real asq = a.dot(&a);
	iris_real bsq = b.dot(&b);
	
	point_t asqb, bsqa;
	asqb.r[0] = asq * b.r[0];
	asqb.r[1] = asq * b.r[1];
	asqb.r[2] = asq * b.r[2];

	bsqa.r[0] = bsq * a.r[0];
	bsqa.r[1] = bsq * a.r[1];
	bsqa.r[2] = bsq * a.r[2];
	
	point_t t1 = asqb.minus(&bsqa);
	point_t axb = a.cross(&b);
	point_t t2 = t1.cross(&axb);
	
	iris_real denom = 2 * axb.dot(&axb);
	t2.r[0] /= denom;
	t2.r[1] /= denom;
	t2.r[2] /= denom;

	point_t c = t2.plus(R+2);
	
	out_ses->c.r[0] = c.r[0];
	out_ses->c.r[1] = c.r[1];
	out_ses->c.r[2] = c.r[2];

	out_ses->r = sqrt(t2.dot(&t2));
    }else if(nr==4) {
	point_t b = R[1].minus(R+0);
	point_t c = R[2].minus(R+0);
	point_t d = R[3].minus(R+0);

	iris_real bsq = b.dot(&b);
	iris_real csq = c.dot(&c);
	iris_real dsq = d.dot(&d);

	point_t b2;
	b2.r[0] = b.r[0] * 2;
	b2.r[1] = b.r[1] * 2;
	b2.r[2] = b.r[2] * 2;

	point_t cxd = c.cross(&d);
	iris_real denom = b2.dot(&cxd);

	point_t dxb = d.cross(&b);
	point_t bxc = b.cross(&c);

	point_t v1;
	v1.r[0] = cxd.r[0] * bsq;
	v1.r[1] = cxd.r[1] * bsq;
	v1.r[2] = cxd.r[2] * bsq;

	point_t v2;
	v2.r[0] = dxb.r[0] * csq;
	v2.r[1] = dxb.r[1] * csq;
	v2.r[2] = dxb.r[2] * csq;

	point_t v3;
	v3.r[0] = bxc.r[0] * dsq;
	v3.r[1] = bxc.r[1] * dsq;
	v3.r[2] = bxc.r[2] * dsq;

	point_t v12 = v1.plus(&v2);
	point_t v123 = v12.plus(&v3);
	point_t i1;
	i1.r[0] = v123.r[0] / denom;
	i1.r[1] = v123.r[1] / denom;
	i1.r[2] = v123.r[2] / denom;

	point_t cc = i1.plus(R+0);
	out_ses->c.r[0] = cc.r[0];
	out_ses->c.r[1] = cc.r[1];
	out_ses->c.r[2] = cc.r[2];

	out_ses->r = sqrt(i1.dot(&i1));
    }
}

static void welzl(point_t *P, int np, point_t *R, int nr, sphere_t *out_ses)
{
    if(np == 0 || nr == 4) {
	sphere_through(R, nr, out_ses);
	return;
    }

    welzl(P+1, np-1, R, nr, out_ses);
    if(out_ses->contains(P[0])) {
	return;
    }

    R[nr] = P[0];
    welzl(P+1, np-1, R, nr+1, out_ses);
}

void ORG_NCSA_IRIS::ses_of_points(point_t *P, int np, sphere_t *out_ses)
{
    point_t R[4];
    
    shuffle(P, np);
    welzl(P, np, R, 0, out_ses);
    if(out_ses->r != 0) {
	out_ses->r *= 1.0001;
    }
}

static bool find_farthest(sphere_t *S, int ns, sphere_t *out_ses, int *out_i)
{
    iris_real max_dist = 0.0;
    int max_idx = 0;
    bool covers_max = false;
    for(int i=0;i<ns;i++) {
	iris_real dx = out_ses->c.r[0] - S[i].c.r[0];
	iris_real dy = out_ses->c.r[1] - S[i].c.r[1];
	iris_real dz = out_ses->c.r[2] - S[i].c.r[2];
	iris_real dist = sqrt(dx*dx + dy*dy + dz*dz);
	if(dist > max_dist) {
	    max_dist = dist;
	    max_idx = i;
	    if(dist + S[i].r <= out_ses->r) {
		covers_max = true;
	    }else {
		covers_max = false;
	    }
	}
    }
    *out_i = max_idx;
    return covers_max;
}

static void enlarge_ses(sphere_t *S, sphere_t *out_ses)
{
    point_t x = S->c.minus(&(out_ses->c));
    iris_real xlen = sqrt(x.dot(&x));
    point_t xhat;
    xhat.r[0] = x.r[0]/xlen;
    xhat.r[1] = x.r[1]/xlen;
    xhat.r[2] = x.r[2]/xlen;
    iris_real R = (S->r + out_ses->r + xlen) * (iris_real)0.5;

    point_t tmp;
    iris_real rr = R - out_ses->r;
    tmp.r[0] = xhat.r[0] * rr;
    tmp.r[1] = xhat.r[1] * rr;
    tmp.r[2] = xhat.r[2] * rr;

    point_t c = out_ses->c.plus(&tmp);
    out_ses->c.r[0] = c.r[0];
    out_ses->c.r[1] = c.r[1];
    out_ses->c.r[2] = c.r[2];
    out_ses->r = R;
}

static void do_sess(sphere_t *S, int ns, sphere_t *out_ses)
{
    if(ns == 0) {
	return;
    }

    // find the farthest from current SES and put it in front
    int i;
    bool covered = find_farthest(S, ns, out_ses, &i);
    sphere_t tmp;
    tmp = S[i];
    S[i] = S[0];
    S[0] = tmp;
    
    if(!covered) {
	enlarge_ses(S, out_ses);
    }
    do_sess(S+1, ns-1, out_ses);
}

void ORG_NCSA_IRIS::ses_of_spheres(sphere_t *S, int ns, sphere_t *out_ses)
{
    if(ns == 0) {
	out_ses->c.r[0] = 0.0;
	out_ses->c.r[1] = 0.0;
	out_ses->c.r[2] = 0.0;
	out_ses->r = 0.0;
	return;
    }

    // start with setting the SES to the first sphere
    out_ses->c.r[0] = S[0].c.r[0];
    out_ses->c.r[1] = S[0].c.r[1];
    out_ses->c.r[2] = S[0].c.r[2];
    out_ses->r = S[0].r;

    do_sess(S+1, ns-1, out_ses);
    
    if(out_ses->r != 0) {
	out_ses->r *= 1.0001;
    }
    
}
