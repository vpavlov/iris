#include "cuda_parameters.h"
using namespace ORG_NCSA_IRIS;

__global__
void copy_to_sendbuf_kernel(iris_real *sendbuf, iris_real ***data,
						int sx, int sy,int sz, int ex,int ey, int ez)
{
    int nx=ex-sx;
    int ny=ey-sy;
    int nz=ez-sz;

    size_t xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    size_t yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    size_t zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = sx+xndx*xchunk_size, i_to = MIN(sx+(xndx+1)*xchunk_size,ex);
	int j_from = sy+yndx*ychunk_size, j_to = MIN(sy+(yndx+1)*ychunk_size,ey);
	int k_from = sz+zndx*zchunk_size, k_to = MIN(sz+(zndx+1)*zchunk_size,ez);

	for(int i=i_from;i<i_to;i++) {
        int ii = i - sx;
        int ni = ii*ny*nz;
		for(int j=j_from;j<j_to;j++) {
            int jj = j - sy;
            int nj = ni + jj*nz;
			for(int k=k_from;k<k_to;k++) {
                int kk = k - sz;
                int n = nj + kk;
				sendbuf[n] = data[i][j][k];
			}
		}
    }
}

void copy_to_sendbuf(iris_real *sendbuf,iris_real ***data,
                    int sx,int sy, int sz, int ex, int ey,int ez)
{
    int nx=ex-sx;
    int ny=ey-sy;
    int nz=ez-sz;

	int nblocks1 = get_NBlocks(nx,IRIS_CUDA_NTHREADS_3D);
	int nblocks2 = get_NBlocks(ny,IRIS_CUDA_NTHREADS_3D);
	int nblocks3 = get_NBlocks(nz,IRIS_CUDA_NTHREADS_3D);
    int nthreads = IRIS_CUDA_NTHREADS_3D;

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads,nthreads,nthreads);
    copy_to_sendbuf_kernel<<<blocks,threads>>>(sendbuf, data, sx, sy, sz, ex, ey, ez);
    cudaDeviceSynchronize();
}

__global__
void copy_to_recvbuf_kernel(iris_real *recvbuf,iris_real ***data, int mode,
                            int sx,int sy, int sz, int ex, int ey,int ez)
{
    int nx=ex-sx;
    int ny=ey-sy;
    int nz=ez-sz;

    size_t xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    size_t yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    size_t zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = sx+xndx*xchunk_size, i_to = MIN(sx+(xndx+1)*xchunk_size,ex);
	int j_from = sy+yndx*ychunk_size, j_to = MIN(sy+(yndx+1)*ychunk_size,ey);
	int k_from = sz+zndx*zchunk_size, k_to = MIN(sz+(zndx+1)*zchunk_size,ez);

	for(int i=i_from;i<i_to;i++) {
        int ii = i - sx;
        int ni = ii*ny*nz;
		for(int j=j_from;j<j_to;j++) {
            int jj = j - sy;
            int nj = ni + jj*nz;
			for(int k=k_from;k<k_to;k++) {
                int kk = k - sz;
                int n = nj + kk;
				if(mode == 0) {
                    data[i][j][k] += recvbuf[n];
                }else {
                    data[i][j][k] = recvbuf[n];
                }
			}
		}
    }
}

void copy_from_recvbuf(iris_real *recvbuf,iris_real ***data, int mode,
    int sx,int sy, int sz, int ex, int ey,int ez)
{
    int nx=ex-sx;
    int ny=ey-sy;
    int nz=ez-sz;
    
	int nblocks1 = get_NBlocks(nx,IRIS_CUDA_NTHREADS_3D);
	int nblocks2 = get_NBlocks(ny,IRIS_CUDA_NTHREADS_3D);
	int nblocks3 = get_NBlocks(nz,IRIS_CUDA_NTHREADS_3D);
    int nthreads = IRIS_CUDA_NTHREADS_3D;

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads,nthreads,nthreads);
    copy_to_recvbuf_kernel<<<blocks,threads>>>(recvbuf, data, mode, sx, sy, sz, ex, ey, ez);
    cudaDeviceSynchronize();
}