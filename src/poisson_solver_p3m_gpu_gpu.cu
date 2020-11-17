#include "cuda_parameters.h"
#include "poisson_solver_p3m_gpu.h"

using namespace ORG_NCSA_IRIS;

__global__
void kspace_phi_kernel(iris_real *io_rho_phi, iris_real *m_greenfn, int nx, int ny, int nz, iris_real scinv)
{
    size_t xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    size_t yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    size_t zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,nx);
	int j_from = yndx*ychunk_size, j_to = MIN((yndx+1)*ychunk_size,ny);
	int k_from = zndx*zchunk_size, k_to = MIN((zndx+1)*zchunk_size,nz);

	for(int i=i_from;i<i_to;i++) {
        int ni = i*ny*nz;
		for(int j=j_from;j<j_to;j++) {
            int nj = ni + j*nz;
			for(int k=k_from;k<k_to;k++) {
                int n = nj + k;

                io_rho_phi[2*n] *= scinv * m_greenfn[n];
                io_rho_phi[2*n+1] *= scinv * m_greenfn[n];
			}
		}
    }
}


void poisson_solver_p3m_gpu::kspace_phi(iris_real *io_rho_phi)
{
    iris_real scaleinv = 1.0/(m_mesh->m_size[0] * m_mesh->m_size[1] * m_mesh->m_size[2]);

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int nblocks1 = get_NBlocks(nx,IRIS_CUDA_NTHREADS_3D);
	int nblocks2 = get_NBlocks(ny,IRIS_CUDA_NTHREADS_3D);
	int nblocks3 = get_NBlocks(nz,IRIS_CUDA_NTHREADS_3D);
    int nthreads = IRIS_CUDA_NTHREADS_3D;

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads,nthreads,nthreads);

    kspace_phi_kernel<<<blocks,threads>>>(io_rho_phi,m_greenfn,nx,ny,nz,scaleinv);
    cudaDeviceSynchronize();
}