// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#include "commrec.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

commrec::commrec(MPI_Comm comm)
{
    m_comm = comm;
    MPI_Comm_size(m_comm, &m_size);
    MPI_Comm_rank(m_comm, &m_rank);
}

commrec::~commrec()
{
}
