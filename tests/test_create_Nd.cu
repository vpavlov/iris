//#include "iris/memory.h"
#include <iris/real.h>
#include <iris/memory.h>
#include <iris/cuda_parameters.h>
#include <stdio.h>
#include <iostream>
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
	int n = i*n2*n3+j*n3+k;	     
	  a[i][j][k] = __int2float_ru(n);
	printf("from %d %d %d to %d %d %d a[%d][%d][%d] = %d %d\n",i_from,j_from,k_from,i_to,j_to,k_to,i,j,k,(int)a[i][j][k],n);	
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
    ORG_NCSA_IRIS::memory_gpu::create_3d(ap,2,3,4,true,777.);

    int nthreads1 = get_NThreads_X(2);
    int nthreads2 = get_NThreads_Y(3);
    int nthreads3 = get_NThreads_Z(4);

    int nblocks1 = get_NBlocks_X(2,nthreads1);
    int nblocks2 = get_NBlocks_Y(3,nthreads2);
    int nblocks3 = get_NBlocks_Z(4,nthreads3);


    auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

	
	std::cout<<"blocks "<<blocks.x<<" "<<blocks.y<<" "<<blocks.z<<std::endl;
	std::cout<<"threads "<<threads.x<<" "<<threads.y<<" "<<threads.z<<std::endl;

    set<<<blocks,threads>>>(ap,2,3,4);
    cudaDeviceSynchronize();
    printf("after set\n");
    print<<<blocks,threads>>>(ap,2,3,4);
    cudaDeviceSynchronize();
    return 0;
}
