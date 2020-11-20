#include "iris/memory.h"
#include <iris/real.h>
#include <iris/memory.h>
#include <iris/cuda_parameters.h>
#include <stdio.h>

__global__
void set(iris_real*** a, int n1, int n2, int n3)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,n1);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,n2);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,n3);

    int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,n1);
    int j_from = yndx*ychunk_size, j_to = MIN((yndx+1)*ychunk_size,n2);
	int k_from = zndx*zchunk_size, k_to = MIN((zndx+1)*zchunk_size,n3);

	for(int i=i_from;i<i_to;i++) {
		for(int j=j_from;j<j_to;j++) {
			for(int k=k_from;k<k_to;k++) {
				a[i][j][k] = 100*i + 10*j + k;
			}
		}
	}
}

__global__
void print(iris_real*** a, int n1, int n2, int n3)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,n1);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,n2);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,n3);

    int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,n1);
    int j_from = yndx*ychunk_size, j_to = MIN((yndx+1)*ychunk_size,n2);
	int k_from = zndx*zchunk_size, k_to = MIN((zndx+1)*zchunk_size,n3);

	for(int i=i_from;i<i_to;i++) {
		for(int j=j_from;j<j_to;j++) {
			for(int k=k_from;k<k_to;k++) {
				printf("a[%d][%d][%d] = %f\n",i,j,k,a[i][j][k]);
			}
		}
	}
}

int main()
{
    iris_real*** ap;
    ORG_NCSA_IRIS::memory_gpu::create_3d(ap,2,3,4,true);

    int nblocks1 = get_NBlocks(2,IRIS_CUDA_NTHREADS_3D);
	int nblocks2 = get_NBlocks(3,IRIS_CUDA_NTHREADS_3D);
	int nblocks3 = get_NBlocks(4,IRIS_CUDA_NTHREADS_3D);
    int nthreads = IRIS_CUDA_NTHREADS_3D;

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
	auto threads = dim3(nthreads,nthreads,nthreads);
	
    set<<<blocks,threads>>>(ap,2,3,4);
    print<<<blocks,threads>>>(ap,2,3,4);
    cudaDeviceSynchronize();
    return 0;
}