// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#ifndef __POISSON_SOLVER_BASE_H__
#define __POISSON_SOLVER_BASE_H__

#include "config.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver_base {
	
    public:
	// Evaluate how good a XxYxZ domain decomposition is.
	// This is solver-dependend (some may prefer slab-style, others
	// cubic, etc.). Larger return results means better configuration.
	// 
	// This is used to figure out how to do automatic domain decomposition
	virtual float eval_dd_conf(int x, int y, int z) = 0;

    };

}
#endif
