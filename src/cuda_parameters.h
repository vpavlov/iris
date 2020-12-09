#pragma once
#include "utils.h"
#include "stdio.h"
#define IRIS_CUDA_NTHREADS_MAX_X 8
#define IRIS_CUDA_NTHREADS_MAX_Y 8
#define IRIS_CUDA_NTHREADS_MAX_Z 8

#define IRIS_CUDA_SHARED_BLOCK_SIZE 128

#define IRIS_CUDA_MAX_NBLOCKS_X 2147483647
#define IRIS_CUDA_MAX_NBLOCKS_Y 65535
#define IRIS_CUDA_MAX_NBLOCKS_Z 65535


#define IRIS_CUDA_INDEX(X) (blockIdx.X*blockDim.X + threadIdx.X)

#define IRIS_CUDA_CHUNK(X,N) ((N+gridDim.X*blockDim.X-1)/(gridDim.X*blockDim.X))

inline
static int get_NBlocks_X(size_t ndata, int nthreads)
{
    return MIN(static_cast<int>((ndata+nthreads-1)/nthreads),IRIS_CUDA_MAX_NBLOCKS_X);
};

inline
static int get_NBlocks_Y(size_t ndata, int nthreads)
{
    return MIN(static_cast<int>((ndata+nthreads-1)/nthreads),IRIS_CUDA_MAX_NBLOCKS_Y);
};

inline
static int get_NBlocks_Z(size_t ndata, int nthreads)
{
    return MIN(static_cast<int>((ndata+nthreads-1)/nthreads),IRIS_CUDA_MAX_NBLOCKS_Z);
};

inline
static int get_NThreads_1D(size_t ndata)
{
    return MIN(ndata,IRIS_CUDA_SHARED_BLOCK_SIZE);
};

inline
static int get_NThreads_X(size_t ndata)
{
    return MIN(ndata,IRIS_CUDA_NTHREADS_MAX_X);
};

inline
static int get_NThreads_Y(size_t ndata)
{
    return MIN(ndata,IRIS_CUDA_NTHREADS_MAX_Y);
};

inline
static int get_NThreads_Z(size_t ndata)
{
    return MIN(ndata,IRIS_CUDA_NTHREADS_MAX_Z);
};

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
	    file, line );
    exit( EXIT_FAILURE );
  }
};

#define HANDLE_LAST_CUDA_ERROR (HandleError(cudaGetLastError(), __FILE__, __LINE__ ))
#define HANDLE_CUDA_ERROR( err ) (HandleError(err, __FILE__, __LINE__ ))