#include "cuda_parameters.h"
#include "poisson_solver_p3m_gpu.h"
#include "mesh_gpu.h"
#include "math_util.h"

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


const int BLOCK_SIZE = IRIS_CUDA_NTHREADS_3D*IRIS_CUDA_NTHREADS_3D*IRIS_CUDA_NTHREADS_3D;

__global__
void kspace_eng_kernel(iris_real *in_rho_phi, iris_real *m_greenfn, iris_real** vc,
                        iris_real *out_Ek_vir,
                        int nx, int ny, int nz, float s2, 
                        int compute_global_energy, int compute_global_virial)
{
    __shared__ iris_real virial_acc[6][BLOCK_SIZE];
    __shared__ iris_real Ek_acc[BLOCK_SIZE];
    
    size_t xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    size_t yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    size_t zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,nx);
	int j_from = yndx*ychunk_size, j_to = MIN((yndx+1)*ychunk_size,ny);
	int k_from = zndx*zchunk_size, k_to = MIN((zndx+1)*zchunk_size,nz);

    int iacc = xndx*IRIS_CUDA_NTHREADS_3D*IRIS_CUDA_NTHREADS_3D + yndx*IRIS_CUDA_NTHREADS_3D + zndx;
    for(int m = 0;m<6;m++) {
    virial_acc[m][iacc] = 0.0;
    }
    Ek_acc[iacc] = 0.0;

    if(compute_global_virial) {
        for(int i=i_from;i<i_to;i++) {
            int ni = i*ny*nz;
            for(int j=j_from;j<j_to;j++) {
                int nj = ni + j*nz;
                for(int k=k_from;k<k_to;k++) {
                int n = nj + k;
                iris_real ener = s2 * m_greenfn[n] *
                (in_rho_phi[2*n  ] * in_rho_phi[2*n  ] +
                 in_rho_phi[2*n+1] * in_rho_phi[2*n+1]);
                for(int m = 0;m<6;m++) {
                    virial_acc[m][iacc] += ener * vc[2*n/2][m];
                }
                if(compute_global_energy) {
                    Ek_acc[iacc] += ener;
                }
                }
            }
        }
    }else {
        for(int i=i_from;i<i_to;i++) {
            int ni = i*ny*nz;
            for(int j=j_from;j<j_to;j++) {
                int nj = ni + j*nz;
                for(int k=k_from;k<k_to;k++) {
                int n = nj + k;
                Ek_acc[iacc] += s2 * m_greenfn[n] *
                (in_rho_phi[2*n  ] * in_rho_phi[2*n  ] +
                 in_rho_phi[2+n+1] * in_rho_phi[2*n+1]);
                }
            }
        }
    }

    __syncthreads();
    
    for(int i = BLOCK_SIZE-1; i > 0; i/=2 ) {
        if (iacc<i && iacc%2==0) {
            for(int m = 0;m<6;m++) {
            virial_acc[m][iacc] += virial_acc[m][iacc+1];
            }
            Ek_acc[iacc] += Ek_acc[iacc+1];
        }
        __syncthreads();
    }

    if (iacc==0) {
        atomicAdd(&(out_Ek_vir[0]),Ek_acc[iacc]);
        for(int m = 0;m<6;m++) {
        atomicAdd(&(out_Ek_vir[m+1]), virial_acc[m][iacc]);
        }
    }
}

void poisson_solver_p3m_gpu::kspace_eng(iris_real *in_rho_phi)
{
    // FFT is not normalized, so we need to do that now
    iris_real s2 = square(1.0/(m_mesh->m_size[0] * m_mesh->m_size[1] * m_mesh->m_size[2]));

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int nthreads=IRIS_CUDA_NTHREADS_3D;

    int nblocks1=static_cast<int>((nx+nthreads-1)/nthreads);
    int nblocks2=static_cast<int>((ny+nthreads-1)/nthreads);
    int nblocks3=static_cast<int>((nz+nthreads-1)/nthreads);

    auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads,nthreads,nthreads);

    kspace_eng_kernel<<<blocks,threads>>>(in_rho_phi, m_greenfn, m_vc, m_Ek_vir, nx, ny, nz, s2, m_iris->m_compute_global_energy, m_iris->m_compute_global_virial);
    cudaDeviceSynchronize();
}
