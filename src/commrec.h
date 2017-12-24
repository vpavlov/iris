// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#ifndef __COMMREC_H__
#define __COMMREC_H__

#include "config.h"
#include <mpi.h>

namespace ORG_NCSA_IRIS {

    class commrec {

    public:

	commrec(MPI_Comm comm);
	~commrec();

	int is_master() { return m_rank == 0; };
	int get_size() { return m_size; };
	int get_rank() { return m_rank; };

    private:
	MPI_Comm m_comm;
	int m_rank;
	int m_size;
    };

}

#endif
