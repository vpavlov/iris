#pragma once
#include "utils.h"
#define IRIS_CUDA_NTHREADS 512
#define IRIS_CUDA_NTHREADS_2D 32
#define IRIS_CUDA_NTHREADS_3D 8

#define IRIS_CUDA_MAX_NBLOCKS 64

#define IRIS_CUDA_INDEX(X) (blockIdx.X*gridDim.X + threadIdx.X)

#define IRIS_CUDA_CHUNK(X,N) (N/(gridDim.X*blockDim.X))

inline
static int get_NBlocks(size_t ndata, int nthreads)
{
    return MAX(static_cast<int>((ndata+nthreads-1)/nthreads),IRIS_CUDA_MAX_NBLOCKS);
}
