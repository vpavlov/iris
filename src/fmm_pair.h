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
#ifndef __IRIS_FMM_PAIR_H__
#define __IRIS_FMM_PAIR_H__

namespace ORG_NCSA_IRIS {

    struct pair_t {
	int sourceID;
	int targetID;

	pair_t(int in_sourceID, int in_targetID)
	{
	    sourceID = in_sourceID;
	    targetID = in_targetID;
	}
    };

    struct pair_comparator_t {
	bool operator()(const pair_t &a, const pair_t &b)
	{
	    if(a.sourceID < b.sourceID) {
		return true;
	    }else if(a.sourceID == b.sourceID) {
		if(a.targetID < b.targetID) {
		    return true;
		}
	    }
	    return false;
	}
    };
	
    struct interact_item_t {
	int sourceID;
	int targetID;
	int ix;
	int iy;
	int iz;

	interact_item_t(int in_sourceID, int in_targetID, int in_ix, int in_iy, int in_iz)
	{
	    sourceID = in_sourceID;
	    targetID = in_targetID;
	    ix = in_ix;
	    iy = in_iy;
	    iz = in_iz;
	};
    };
}

#endif
