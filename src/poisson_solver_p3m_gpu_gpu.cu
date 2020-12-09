#include "cuda_parameters.h"
#include "poisson_solver_p3m_gpu.h"
#include "mesh_gpu.h"
#include "math_util.h"
#include "domain_gpu.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

__global__
void kspace_phi_kernel(iris_real *io_rho_phi, iris_real *m_greenfn, int nx, int ny, int nz, iris_real scinv)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
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
    
    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    kspace_phi_kernel<<<blocks,threads>>>(io_rho_phi,m_greenfn,nx,ny,nz,scaleinv);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}

__global__
void kspace_eng_kernel(iris_real *in_rho_phi, iris_real *m_greenfn, iris_real** vc,
                        iris_real *out_Ek_vir,
                        int nn, float s2, 
                        int compute_global_energy, int compute_global_virial, iris_real u_factor)
{
    __shared__ iris_real virial_acc[6][IRIS_CUDA_SHARED_BLOCK_SIZE];
    __shared__ iris_real Ek_acc[IRIS_CUDA_SHARED_BLOCK_SIZE];
    
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nn);

	int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,nn);

    int iacc = (xndx-blockIdx.x*blockDim.x);

    for(int m = 0;m<6;m++) {
    virial_acc[m][iacc] = 0.0;
    }
    Ek_acc[iacc] = 0.0;

 //printf(" i_from %d i_to %d iacc %d xndx %d \n",i_from,i_to,iacc,xndx);

        //printf("if xndx %d yndx %d zndx %d\n",xndx,yndx,zndx);
        if(compute_global_virial) {
            for(int n=i_from;n<i_to;n++) {
                    iris_real ener = s2 * m_greenfn[n] *
                    (in_rho_phi[2*n  ] * in_rho_phi[2*n  ] +
                    in_rho_phi[2*n+1] * in_rho_phi[2*n+1]);
                    // if(nx*ny*nz-n<10) 
                    //   printf("n %d  m_greenfn[%d] %f ener %f Ek_acc[%d] %f\n",n,n,m_greenfn[n],ener,iacc,Ek_acc[iacc]);
                    for(int m = 0;m<6;m++) {
                        virial_acc[m][iacc] += ener * vc[2*n/2][m];
                    }
                    if(compute_global_energy) {
                        Ek_acc[iacc] += ener;
                    }
                    }
        }else {
            for(int n=i_from;n<i_to;n++) {
                    Ek_acc[iacc] += s2 * m_greenfn[n] *
                    (in_rho_phi[2*n  ] * in_rho_phi[2*n  ] +
                    in_rho_phi[2+n+1] * in_rho_phi[2*n+1]);
                   }
        }

    

    __syncthreads();
    
    for(int i = IRIS_CUDA_SHARED_BLOCK_SIZE; i > 0; i/=2 ) {
        int stride = IRIS_CUDA_SHARED_BLOCK_SIZE/i;
        if (iacc < (IRIS_CUDA_SHARED_BLOCK_SIZE - stride)  && (iacc)%(2*stride)==0) {
            for(int m = 0;m<6;m++) {
                virial_acc[m][iacc] += virial_acc[m][iacc+stride];
            }
            Ek_acc[iacc] += Ek_acc[iacc+stride];
            // if(stride>1&&stride%2)
            
        }
       // printf("i %d iacc %d stride %d iacc+stride %d \n",i,iacc,stride,iacc+stride);
        __syncthreads();
    }

    if (iacc==0) {
        atomicAdd(&(out_Ek_vir[0]),Ek_acc[iacc]*u_factor);
        for(int m = 0;m<6;m++) {
        atomicAdd(&(out_Ek_vir[m+1]), virial_acc[m][iacc]*u_factor);
        }
    }
    // if (xndx==0) {
    //     printf(" out_Ek_vir[0] %f\n",  out_Ek_vir[0]);
    //     out_Ek_vir[0] *= u_factor;
    //     for(int m = 0;m<6;m++) {
    //     out_Ek_vir[m+1]*=u_factor;
    //     }
    //     printf(" out_Ek_vir[0]*u_factor %f\n",  out_Ek_vir[0]);
    // }
}

void poisson_solver_p3m_gpu::kspace_eng(iris_real *in_rho_phi)
{
    // FFT is not normalized, so we need to do that now
    iris_real s2 = square(1.0/(m_mesh->m_size[0] * m_mesh->m_size[1] * m_mesh->m_size[2]));

    iris_real post_corr = 0.5 *
	m_domain->m_global_box.xsize *
	m_domain->m_global_box.ysize *
	m_domain->m_global_box.zsize *
	m_units->ecf;

    if(m_iris->m_compute_global_energy||m_iris->m_compute_global_virial)
    {
        cudaMemset(m_iris->m_Ek_vir,0,7*sizeof(iris_real));
    }

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    //int nthreads=IRIS_CUDA_NTHREADS_3D;

    // the kernel has to be rewritten in move convenient way

    int nthreads1 = IRIS_CUDA_SHARED_BLOCK_SIZE;
	// int nthreads3 = IRIS_CUDA_NTHREADS_Z;
    int nblocks1 = get_NBlocks_X(nx*ny*nz,IRIS_CUDA_SHARED_BLOCK_SIZE);
	// int nblocks3 = get_NBlocks_Z(nz,IRIS_CUDA_NTHREADS_Z);

	auto blocks = dim3(nblocks1);
    auto threads = dim3(nthreads1);

printf("bl %d %d %d the %d %d %d\n",blocks.x,blocks.y,blocks.z,threads.x,threads.y,threads.z);

    kspace_eng_kernel<<<blocks,threads>>>(in_rho_phi, m_greenfn, m_vc, m_iris->m_Ek_vir, nx*ny*nz, s2, m_iris->m_compute_global_energy, m_iris->m_compute_global_virial,post_corr);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    
   // m_mesh->dump_ascii_from_gpu("denominator_x",m_denominator_x,1,1,m_fft_size[0]);
   // m_mesh->dump_ascii_from_gpu("denominator_y",m_denominator_y,1,1,m_fft_size[1]);
   // m_mesh->dump_ascii_from_gpu("denominator_z",m_denominator_z,1,1,m_fft_size[2]);
   // m_mesh->dump_ascii_from_gpu("greenfn",m_greenfn,1,1,m_fft_size[0]*m_fft_size[1]*m_fft_size[2]);
	// exit(333);
   //  exit(777);
}


__global__
void kspace_Ex_kernel(iris_real *in_phi, iris_real *out_Ex, iris_real *kx, int nx, int ny, int nz)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
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

                out_Ex[2*n] = in_phi[2*n+1] * kx[i];
                out_Ex[2*n+1] = -in_phi[2*n] * kx[i]; 
            }
		}
    }
}

void poisson_solver_p3m_gpu::kspace_Ex(iris_real *in_phi, iris_real *out_Ex)
{
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    kspace_Ex_kernel<<<blocks,threads>>>(in_phi, out_Ex, m_kx, nx, ny, nz);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}

__global__
void kspace_Ey_kernel(iris_real *in_phi, iris_real *out_Ey, iris_real *ky, int nx, int ny, int nz)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
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

                out_Ey[2*n] = in_phi[2*n+1] * ky[j];
                out_Ey[2*n+1] = -in_phi[2*n] * ky[j];
			}
		}
    }
}

void poisson_solver_p3m_gpu::kspace_Ey(iris_real *in_phi, iris_real *out_Ey)
{
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    kspace_Ey_kernel<<<blocks,threads>>>(in_phi, out_Ey, m_ky, nx, ny, nz);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}

__global__
void kspace_Ez_kernel(iris_real *in_phi, iris_real *out_Ez, iris_real *kz, int nx, int ny, int nz)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
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

                out_Ez[2*n] = in_phi[2*n+1] * kz[k];
                out_Ez[2*n+1] = -in_phi[2*n] * kz[k];
            }
		}
    }
}

void poisson_solver_p3m_gpu::kspace_Ez(iris_real *in_phi, iris_real *out_Ez)
{
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];

    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    kspace_Ez_kernel<<<blocks,threads>>>(in_phi, out_Ez, m_kz, nx, ny, nz);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}

__device__
inline iris_real denominator1_dev(const iris_real &x, int order, iris_real *gfd_coeff)
{
    iris_real sx;
    sx = 0.0;

    for(int i = order - 1; i >= 0; i--) {
        iris_real c = gfd_coeff[i];
        sx = c + sx * x;
    }
    
    return sx*sx;
}

__device__
static inline iris_real square_dev(const iris_real &x) { return x*x;}

__device__
static inline iris_real pow_sinx_x_dev(const double &x, int n)
{
    if (x == 0.0) {
        return 1.0;
    }

    iris_real sinx_x = sin(x)/x;
    iris_real retval = 1.0;
    while(n != 0) {
        if (n & 1) {
        retval *= sinx_x;
        }
        sinx_x *= sinx_x;
        n >>= 1;
    }

    return retval;
}

#define  _PI   3.141592653589793238462643383279
#define _2PI   6.283185307179586476925286766559
#define _4PI  12.56637061435917295385057353311
#define  EPS   1.0e-7

__global__
void calculate_denominator_kernel(iris_real *denominator_r, int sr, int nr, int rM, int order, iris_real *gfd_coeff)
{
    int rndx = IRIS_CUDA_INDEX(x);
    int rchunk_size = IRIS_CUDA_CHUNK(x,nr);

    int r_from = rndx*rchunk_size, r_to = MIN((rndx+1)*rchunk_size,nr);
    
	for (int r = sr + r_from; r < sr + r_to; r++) {
	    int rj = r - rM * (2 * r / rM);
	    iris_real sinr2 = square_dev(sin(_PI * rj / rM));
	    denominator_r[r - sr] = denominator1_dev(sinr2,order,gfd_coeff);
	}
}


void poisson_solver_p3m_gpu::calculate_denominator()
{
    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];

    int nx = m_fft_size[0];
	int ny = m_fft_size[1];
	int nz = m_fft_size[2];
	    
	int sx = m_fft_offset[0];
	int sy = m_fft_offset[1];
	int sz = m_fft_offset[2];
    
    int nthreads = get_NThreads_X(nx);
    int nblocks = get_NBlocks_X(nx,nthreads);

    calculate_denominator_kernel<<<nblocks,nthreads>>>(m_denominator_x, sx, nx, xM, m_chass->m_order, m_chass->m_gfd_coeff);

    nthreads = get_NThreads_X(ny);
    nblocks = get_NBlocks_X(ny,nthreads);
    calculate_denominator_kernel<<<nblocks,nthreads>>>(m_denominator_y, sy, ny, yM, m_chass->m_order, m_chass->m_gfd_coeff);

    nthreads = get_NThreads_X(nz);
    nblocks = get_NBlocks_X(nz,nthreads);
    calculate_denominator_kernel<<<nblocks,nthreads>>>(m_denominator_z, sz, nz, zM, m_chass->m_order, m_chass->m_gfd_coeff);

    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}

__global__
void calculate_gf_fact_1_kernel(iris_real *greenfn_r, iris_real *denominator_r, int sr, int nr, int rM, iris_real krm, int _2n, iris_real rL, iris_real alpha)
{
    int rndx = IRIS_CUDA_INDEX(x);
    int rchunk_size = IRIS_CUDA_CHUNK(x,nr);

    int r_from = rndx*rchunk_size, r_to = MIN((rndx+1)*rchunk_size,nr);

    for (int r = sr + r_from; r < sr + r_to; r++) {
	    int rj = r - rM * (2 * r / rM);
	    iris_real rkplusb = krm * rj;
	    iris_real rrho = exp(-0.25 * square_dev(rkplusb / alpha));
	    iris_real rwnsq = pow_sinx_x_dev(rkplusb * rL / (2 * rM), _2n);
	    iris_real part2 = rrho * rwnsq;
	    greenfn_r[r - sr] = part2 / denominator_r[r - sr];
//        printf("greenfn_r[%d] %f rkplusb %f rrho %f rwnsq %f rj %d krm %f\n",r - sr,greenfn_r[r - sr],rkplusb, rrho, rwnsq, rj,krm );
	}

}

__global__
void calculate_gf_fact_2_kernel(iris_real *greenfn, 
    iris_real *greenfn_x, iris_real *greenfn_y, iris_real *greenfn_z,
    int sx, int sy, int sz, int ex, int ey, int ez,
    int xM, int yM, int zM,
    iris_real kxm, iris_real kym, iris_real kzm)
{
    int nx = ex - sx;
    int ny = ey - sy;
    int nz = ez - sz;

    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = sx + xndx*xchunk_size, i_to = MIN(sx+(xndx+1)*xchunk_size,ex);
	int j_from = sy + yndx*ychunk_size, j_to = MIN(sy+(yndx+1)*ychunk_size,ey);
    int k_from = sz + zndx*zchunk_size, k_to = MIN(sz+(zndx+1)*zchunk_size,ez);
    
    for (int x = i_from; x < i_to; x++) {
	    int xj = x - xM * (2 * x / xM);
	    for (int y = j_from; y < j_to; y++) {
		int yj = y - yM * (2 * y / yM);
		for (int z = k_from; z < k_to; z++) {
		    int zj = z - zM * (2 * z / zM); // convert from 0..P to 0..P/2, -P/2...-1
		    iris_real ksq = square_dev(kxm * xj) + square_dev(kym * yj) + square_dev(kzm * zj);
		    if (ksq != 0.0) {
			iris_real part1 = _4PI / ksq;
			iris_real part2 = greenfn_x[x - sx] * greenfn_y[y - sy] * greenfn_z[z - sz];
			greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = part1 * part2;
		    }else {
			greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = 0.0;
		    }
		}
	    }
	}
}

void poisson_solver_p3m_gpu::calculate_gf_fact()
{
    const iris_real alpha = m_iris->m_alpha;

    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];
    
    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);
	
    const int _2n = 2*m_chass->m_order;

    iris_real *greenfn_x = NULL;
    iris_real *greenfn_y = NULL;
    iris_real *greenfn_z = NULL;
	
    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;
   // printf("sx %d nx %d xM %d kxm %f _2n %d xL %d alpha %f\n", sx, nx, xM, kxm, _2n, xL, alpha);
    memory_gpu::create_1d(greenfn_x, nx);
    memory_gpu::create_1d(greenfn_y, ny);
    memory_gpu::create_1d(greenfn_z, nz);

    int nthreads = get_NThreads_X(nx);
    int nblocks = get_NBlocks_X(nx,nthreads);

    calculate_gf_fact_1_kernel<<<nblocks,nthreads>>>(greenfn_x, m_denominator_x, sx, nx, xM, kxm, _2n, xL, alpha);

    nthreads = get_NThreads_X(ny);
    nblocks = get_NBlocks_X(ny,nthreads);
    calculate_gf_fact_1_kernel<<<nblocks,nthreads>>>(greenfn_y, m_denominator_y, sy, ny, yM, kym, _2n, yL, alpha);

    nthreads = get_NThreads_X(nz);
    nblocks = get_NBlocks_X(nz,nthreads);
    calculate_gf_fact_1_kernel<<<nblocks,nthreads>>>(greenfn_z, m_denominator_z, sz, nz, zM, kzm, _2n, zL, alpha);

    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;

    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    calculate_gf_fact_2_kernel<<<blocks,threads>>>(
        m_greenfn, greenfn_x, greenfn_y, greenfn_z,
                sx, sy, sz, ex, ey, ez,
                xM, yM, zM,
                kxm, kym, kzm);
                
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    
    memory_gpu::destroy_1d(greenfn_x);
    memory_gpu::destroy_1d(greenfn_y);
    memory_gpu::destroy_1d(greenfn_z);
}


__global__
void calculate_gf_full_kernel(iris_real *greenfn, 
    iris_real *denominator_x, iris_real *denominator_y, iris_real *denominator_z,
    int sx, int sy, int sz, int ex, int ey, int ez,
    int xM, int yM, int zM,
    iris_real xL,  iris_real yL,  iris_real zL,
    iris_real kxm, iris_real kym, iris_real kzm,
    int nbx, int nby, int nbz, iris_real alpha, int _2n)
{
    int nx = ex - sx;
    int ny = ey - sy;
    int nz = ez - sz;

    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = sx + xndx*xchunk_size, i_to = MIN(sx+(xndx+1)*xchunk_size,ex);
	int j_from = sy + yndx*ychunk_size, j_to = MIN(sy+(yndx+1)*ychunk_size,ey);
    int k_from = sz + zndx*zchunk_size, k_to = MIN(sz+(zndx+1)*zchunk_size,ez);
    
    for (int x = i_from; x < i_to; x++) {
	    int xj = x - xM * (2 * x / xM);
	    for (int y = j_from; y < j_to; y++) {
            int yj = y - yM * (2 * y / yM);
            for (int z = k_from; z < k_to; z++) {
                int zj = z - zM * (2 * z / zM); // convert from 0..P to 0..P/2, -P/2...-1
                //////////////////////////////////////////////////
                iris_real ksq = square_dev(kxm * xj) + square_dev(kym * yj) + square_dev(kzm * zj);
                if (ksq != 0.0) {
                iris_real part1 = _4PI / ksq;
                iris_real part2 = 0.0;
                    for (int bx = -nbx; bx <= nbx; bx++) {
                        iris_real xkplusb = kxm * (xj + xM * bx);
                        iris_real xrho = exp(-0.25 * square_dev(xkplusb / alpha));
                        iris_real xwnsq = pow_sinx_x_dev(xkplusb * xL / (2 * xM), _2n);

                        for (int by = -nby; by <= nby; by++) {
                        iris_real ykplusb = kym * (yj + yM * by);
                        iris_real yrho = exp(-0.25 * square_dev(ykplusb / alpha));
                        iris_real ywnsq = pow_sinx_x_dev(ykplusb * yL / (2 * yM), _2n);
                                        
                            for (int bz = -nbz; bz <= nbz; bz++) {
                                iris_real zkplusb = kzm * (zj + zM * bz);
                                iris_real zrho = exp(-0.25 * square_dev(zkplusb / alpha));
                                iris_real zwnsq = pow_sinx_x_dev(zkplusb * zL / (2 * zM), _2n);
                                
                                // k . (k+b)
                                iris_real k_dot_kplusb = kxm * xj * xkplusb + kym * yj * ykplusb + kzm * zj * zkplusb;
                                                
                                // (k+b) . (k+b)
                                iris_real kplusb_sq = xkplusb * xkplusb + ykplusb * ykplusb + zkplusb * zkplusb;
                                                
                                part2 += (k_dot_kplusb / kplusb_sq) * xrho * yrho * zrho * xwnsq * ywnsq * zwnsq;
                            }
                        }
                    }
                    iris_real part3 = denominator_x[x - sx]*denominator_y[y - sy]*denominator_z[z - sz];
                    greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = part1 * part2 / part3;
                }else {
                    greenfn[ROW_MAJOR_OFFSET(x-sx, y-sy, z-sz, ny, nz)] = 0.0;
                }
            }
        }
    }
}

void poisson_solver_p3m_gpu::calculate_gf_full()
{
    const iris_real alpha = m_iris->m_alpha;

    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];

    const int nbx = static_cast<int> ((alpha*xL/(_PI*xM)) * pow(-log(EPS),0.25));
    const int nby = static_cast<int> ((alpha*yL/(_PI*yM)) * pow(-log(EPS),0.25));
    const int nbz = static_cast<int> ((alpha*zL/(_PI*zM)) * pow(-log(EPS),0.25));
    
    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);
	
    const int _2n = 2*m_chass->m_order;

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    calculate_gf_full_kernel<<<blocks,threads>>>(m_greenfn, 
                            m_denominator_x, m_denominator_y, m_denominator_z,
                            sx, sy, sz, ex, ey, ez,
                            xM, yM, zM,
                            xL,   yL,   zL,
                            kxm,  kym,  kzm,
                            nbx, nby, nbz,  alpha, _2n);

    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}

__global__
void calculate_k_kernel(iris_real *kr, iris_real krm, int sr, int er, int rM)
{
    int nr = er - sr;

    int rndx = IRIS_CUDA_INDEX(x);
    int rchunk_size = IRIS_CUDA_CHUNK(x,nr);

    int r_from = sr + rndx*rchunk_size, r_to = MIN(sr+(rndx+1)*rchunk_size,er);

    for(int r = r_from; r < r_to; r++) {
        int rj = r - rM*(2*r/rM);
        kr[r-sr] = krm * rj;
    }
}

void poisson_solver_p3m_gpu::calculate_k()
{
    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const iris_real kxm = (_2PI/xL);
    const iris_real kym = (_2PI/yL);
    const iris_real kzm = (_2PI/zL);

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    int nthreads = get_NThreads_X(nx);;
    int nblocks = get_NBlocks_X(nx,nthreads);
    calculate_k_kernel<<<nblocks,nthreads>>>(m_kx, kxm, sx, ex, xM);

    nthreads = get_NThreads_X(ny);
    nblocks = get_NBlocks_X(ny,nthreads);
    calculate_k_kernel<<<nblocks,nthreads>>>(m_ky, kym, sy, ey, yM);

    nthreads = get_NThreads_X(nz);
    nblocks = get_NBlocks_X(nz,nthreads);
    calculate_k_kernel<<<nblocks,nthreads>>>(m_kz, kzm, sz, ez, zM);

    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}


__global__
void calculate_virial_coeff_kernel(iris_real **m_vc, 
                            iris_real *m_kx, iris_real *m_ky, iris_real *m_kz, 
                            int sx, int sy, int sz, int ex, int ey, int ez, iris_real alpha)
{
    int nx = ex - sx;
    int ny = ey - sy;
    int nz = ez - sz;

    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = sx + xndx*xchunk_size, i_to = MIN(sx+(xndx+1)*xchunk_size,ex);
	int j_from = sy + yndx*ychunk_size, j_to = MIN(sy+(yndx+1)*ychunk_size,ey);
    int k_from = sz + zndx*zchunk_size, k_to = MIN(sz+(zndx+1)*zchunk_size,ez);
    
    for(int x = i_from; x < i_to; x++) {
        int ni = (x-sx)*ny*nz;
    for(int y = j_from; y < j_to; y++) {
        int nj = ni + (y-sy)*nz;
        for(int z = k_from; z < k_to; z++) {
            int n = nj + z-sz;
        iris_real sq =
            m_kx[x-sx]*m_kx[x-sx] +
            m_ky[y-sy]*m_ky[y-sy] +
            m_kz[z-sz]*m_kz[z-sz];
        if(sq == 0.0) {
            m_vc[n][0] = m_vc[n][1] = m_vc[n][2] =
            m_vc[n][3] = m_vc[n][4] = m_vc[n][5] = 0.0;
        }else {
            iris_real t = -2.0/sq - 0.5/(alpha * alpha);
            m_vc[n][0] = 1.0 + t * m_kx[x-sx] * m_kx[x-sx];
            m_vc[n][1] = 1.0 + t * m_ky[y-sy] * m_ky[y-sy];
            m_vc[n][2] = 1.0 + t * m_kz[z-sz] * m_kz[z-sz];
            m_vc[n][3] = t * m_kx[x-sx] * m_ky[y-sy];
            m_vc[n][4] = t * m_kx[x-sx] * m_kz[z-sz];
            m_vc[n][5] = t * m_ky[y-sy] * m_kz[z-sz];
        }
        }
    }
    }
}

void poisson_solver_p3m_gpu::calculate_virial_coeff()
{
    const iris_real alpha = m_iris->m_alpha;

    int nx = m_fft_size[0];
    int ny = m_fft_size[1];
    int nz = m_fft_size[2];
    
    int sx = m_fft_offset[0];
    int sy = m_fft_offset[1];
    int sz = m_fft_offset[2];
    
    int ex = sx + nx;
    int ey = sy + ny;
    int ez = sz + nz;

    int nthreads1 = get_NThreads_X(nx);
	int nthreads2 = get_NThreads_Y(ny);
	int nthreads3 = get_NThreads_Z(nz);
    int nblocks1 = get_NBlocks_X(nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);

    calculate_virial_coeff_kernel<<<blocks,threads>>>(m_vc, m_kx, m_ky, m_kz, sx, sy, sz, ex, ey, ez, alpha);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}
