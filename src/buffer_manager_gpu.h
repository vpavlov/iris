// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2020, the National Center for Supercomputing Applications
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
#ifndef __IRIS_GPU_BUFFER_MANAGER_H__
#define __IRIS_GPU_BUFFER_MANAGER_H__
#include <limits>
#include <new>

#include "memory.h"

namespace ORG_NCSA_IRIS {

    template<typename T>
    class buffer_manager_gpu {
        private:
        int m_max_allocated_nbytes;
        int m_max_nbuffers;
        int m_nbytes_allocated;
        bool* m_buffer_in_use;
        int* m_buffer_nbytes;
        T** m_buffers;
        public:
        buffer_manager_gpu(int max_nbuffers, int max_allocated_nbytes):
						m_max_nbuffers(max_nbuffers), 
						m_max_allocated_nbytes(max_allocated_nbytes),
						m_nbytes_allocated(0) 
        {
        m_buffers = new T*[m_max_nbuffers];
        m_buffer_in_use = new bool[m_max_nbuffers];
        m_buffer_nbytes = new int[m_max_nbuffers];
        for(int ii=0;ii!=m_max_nbuffers;++ii) {
            m_buffers[ii] = nullptr;
            m_buffer_in_use[ii] = false;
            m_buffer_nbytes[ii] = 0;
        }
        };
        virtual ~buffer_manager_gpu()
        {
        delete []m_buffers;
        delete []m_buffer_in_use;
        for(int ii=0;ii!=m_max_nbuffers;++ii) {
            memory_gpu::wfree(m_buffers[ii]);
        }
        };
        void release_buffer(T* buffer)
        {
        for(int ii=0;ii!=m_max_nbuffers;++ii) {
            if (buffer==m_buffers[ii]) {
                m_buffer_in_use[ii] = false;
                return;
            }
        }
        };
        T* take_buffer(int nbytes)
        {
        int the_smallest_buffer_index=0;
        int the_smallest_buffer_nbytes=std::numeric_limits<int>::max();
        int nbuffers_in_use=0;
        for(int ii=0;ii!=m_max_nbuffers;++ii) {
            if (!m_buffer_in_use[ii]){
                if (m_buffer_nbytes[ii]==0) {
                    m_buffers[ii] = static_cast<T*>(ORG_NCSA_IRIS::memory_gpu::wmalloc(nbytes));
                    m_buffer_nbytes[ii] = nbytes;
                    if ( m_nbytes_allocated + nbytes > m_max_allocated_nbytes ) {
                        throw "total allocated buffers size exceeded...";
                    }
                    m_nbytes_allocated += nbytes;
                    m_buffer_in_use[ii]=true;
                    return m_buffers[ii];
                } else if (m_buffer_nbytes[ii]>=nbytes) {
                    m_buffer_in_use[ii]=true;
                    return m_buffers[ii];
                }
                if(m_buffer_nbytes[ii]<the_smallest_buffer_nbytes) {
                    the_smallest_buffer_index=ii;
                    the_smallest_buffer_nbytes=m_buffer_nbytes[ii];
                }
            } else {
                nbuffers_in_use++;
            }
        }
        if (nbuffers_in_use == m_max_nbuffers) {
            throw "Maximum number of buffers reached...";
        }
        if (m_nbytes_allocated-the_smallest_buffer_nbytes+nbytes>m_max_allocated_nbytes){
            throw "the buffer requested size cannot be allocated";
        }
        int ii = the_smallest_buffer_index;
        m_buffers[ii] = static_cast<T*>(memory_gpu::wrealloc(m_buffers[ii],nbytes,m_buffer_nbytes[ii]));
        m_buffer_in_use[ii] = true;
        m_nbytes_allocated-=m_buffer_nbytes[ii];
        m_nbytes_allocated+=nbytes;
        m_buffer_nbytes[ii]=nbytes;
        return m_buffers[ii];
        };
        void free_buffer(T* buffer)
        {
        for(int ii=0;ii!=m_max_nbuffers;++ii) {
            if (buffer==m_buffers[ii]) {
                m_buffer_in_use[ii] = false;
                m_nbytes_allocated -= m_buffer_nbytes[ii];
                m_buffer_nbytes[ii] = 0;
                memory_gpu::wfree(m_buffers[ii]);
                return;
            }
        }   
        };
    };

};

#endif