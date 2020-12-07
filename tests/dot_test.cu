#include <iris/cuda_parameters.h>
//#include <box.h>
const int BLOCK_SIZE=IRIS_CUDA_NTHREADS_Z*IRIS_CUDA_NTHREADS_Z*IRIS_CUDA_NTHREADS_Z;

using namespace ORG_NCSA_IRIS;

__global__
void kspace_eng_kernel(iris_real *in_rho_phi,
                        iris_real *out_Ek_vir,
                        int nx, int ny, int nz, float s2, 
                        int compute_global_energy, int compute_global_virial)
{
    __shared__ iris_real virial_acc[6][BLOCK_SIZE];
    __shared__ iris_real Ek_acc[BLOCK_SIZE];
    
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,ny);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,nz);

	int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,nx);
	int j_from = yndx*ychunk_size, j_to = MIN((yndx+1)*ychunk_size,ny);
	int k_from = zndx*zchunk_size, k_to = MIN((zndx+1)*zchunk_size,nz);

    int iacc = xndx*IRIS_CUDA_NTHREADS_Z*IRIS_CUDA_NTHREADS_Z + yndx*IRIS_CUDA_NTHREADS_Z + zndx;



    // printf("bdimx %d bdimy %d bdimz %d tidx %d tidy %d tidz %d\n",blockDim.x,blockDim.y,blockDim.z, threadIdx.x, threadIdx.y,threadIdx.z);

    //printf("iacc = %d i_from %d i_to %d j_from %d j_to %d k_from %d k_to %d xchunk_size %d ychunk_size %d zchunk_size %d nx %d ny %d nz %d\n",iacc,i_from,i_to,j_from,j_to,k_from,k_to, xchunk_size, ychunk_size, zchunk_size,nx,ny,nz);
    //printf("out of the for bdimx %d bdimy %d bdimz %d tidx %d tidy %d tidz %d ndx %d %d %d i_from %d i_to %d j_from %d j_to %d k_from %d k_to %d\n",blockDim.x,blockDim.y,blockDim.z, threadIdx.x, threadIdx.y,threadIdx.z,xndx,yndx,zndx,i_from,i_to,j_from,j_to,k_from,k_to);
   
    
    for(int m = 0;m<6;m++) {
    virial_acc[m][iacc] = 0.0;
    }
    Ek_acc[iacc] = 0.0;

    
    for(int i=i_from;i<i_to;i++) {
        int ni = i*ny*nz;
        for(int j=j_from;j<j_to;j++) {
            int nj = ni + j*nz;
            for(int k=k_from;k<k_to;k++) {
               //printf("bdimx %d bdimy %d bdimz %d tidx %d tidy %d tidz %d ndx %d %d %d\n",blockDim.x,blockDim.y,blockDim.z, threadIdx.x, threadIdx.y,threadIdx.z,xndx,yndx,zndx);
               // printf("echo i %d j %d k %d gridDim.x %d gridDim.y %d gridDim.z %d\n", i,j,k,gridDim.x,gridDim.y,gridDim.z );
            int n = nj + k;
            //in_rho_phi[2*n  ] = 1.0; // n*1.0;
            //in_rho_phi[2*n+1] = 1.0; //(n+1)*1.0;
            }
        }
    }

    if (xndx<nx || yndx<ny || zndx < nz)
    {

  // printf("bdimx %d bdimy %d bdimz %d tidx %d tidy %d tidz %d\n",blockDim.x,blockDim.y,blockDim.z, threadIdx.x, threadIdx.y,threadIdx.z);

    if(compute_global_virial) {
        for(int i=i_from;i<i_to;i++) {
            int ni = i*ny*nz;
            for(int j=j_from;j<j_to;j++) {
                int nj = ni + j*nz;
                for(int k=k_from;k<k_to;k++) {
                int n = nj + k;
               // printf("bdimx %d bdimy %d bdimz %d tidx %d tidy %d tidz %d i %d j %d k %d in_rho_phi[%d] %f in_rho_phi[%d] %f\n",blockDim.x,blockDim.y,blockDim.z,threadIdx.x, threadIdx.y,threadIdx.z,i,j,k,2*n,in_rho_phi[2*n  ],2*n+1, in_rho_phi[2*n+1]);
                continue;
                iris_real ener = s2 *
                (in_rho_phi[2*n  ] * in_rho_phi[2*n  ] +
                 in_rho_phi[2*n+1] * in_rho_phi[2*n+1]);
                for(int m = 0;m<6;m++) {
                    virial_acc[m][iacc] += ener;
                }
                if(compute_global_energy) {
                    Ek_acc[iacc] += ener;
                }
                //printf("echo i %d j %d k %d n %d Ek_acc[%d] %f\n",i,j,k,n, iacc,Ek_acc[iacc]);
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
                Ek_acc[iacc] += s2 *
                (in_rho_phi[2*n  ] * in_rho_phi[2*n  ] +
                 in_rho_phi[2+n+1] * in_rho_phi[2*n+1]);
                }
            }
        }
    }
}
    __syncthreads();

    for(int i = BLOCK_SIZE; i > 1; i/=2 ) {
        //printf("echo  BLOCK_SIZE %d ibl %d iacc %d + %d BLOCK_SIZE/i %d (iacc)%(BLOCK_SIZE/i) %d \n",BLOCK_SIZE,i,iacc,iacc+BLOCK_SIZE/i,BLOCK_SIZE/i,(iacc)%(BLOCK_SIZE/i));
        int stride = BLOCK_SIZE/i;
        if (iacc < (BLOCK_SIZE - stride)  && (iacc)%(2*stride)==0) {
          //  printf("i %d Ek_acc[%d] %f Ek_acc[%d] %f\n", i, iacc,Ek_acc[iacc],iacc+stride,Ek_acc[iacc+stride]);
            for(int m = 0;m<6;m++) {
                virial_acc[m][iacc] += virial_acc[m][iacc+stride];
            }
            Ek_acc[iacc] += Ek_acc[iacc+stride];
            //printf("echo ibl %d iacc %d + %d BLOCK_SIZE/i %d (iacc)%(BLOCK_SIZE/i) %d \n",i,iacc,iacc+BLOCK_SIZE/i,BLOCK_SIZE/i,(iacc)%(BLOCK_SIZE/i));
            
       }
        __syncthreads();
    }

    if (iacc==0) {
        atomicAdd(&(out_Ek_vir[0]),Ek_acc[iacc]);
        for(int m = 0;m<6;m++) {
        atomicAdd(&(out_Ek_vir[m+1]), virial_acc[m][iacc]);
        }
        for(int kk=0; kk<7;kk++){
            //printf("out_Ek_vir[%d] -> %f\n",kk,out_Ek_vir[kk]);
        }
    }
}

int main()
{
    iris_real *in_rho_phi;
    iris_real *out_Ek_vir;
    int nx = 32, ny =32, nz = 32;
    float s2 = 1; 
    int compute_global_energy = 1;
    int compute_global_virial = 1;

    cudaMalloc(&in_rho_phi, 2*nx*ny*nz);
    cudaMalloc(&out_Ek_vir, 7);

    //cudaMemset(in_rho_phi,1.0,2*nx*ny*nz);

    int nthreads=IRIS_CUDA_NTHREADS_Z;

    int nblocks1=1;//static_cast<int>((nx+nthreads-1)/nthreads);
    int nblocks2=1;//static_cast<int>((ny+nthreads-1)/nthreads);
    int nblocks3=1;//static_cast<int>((nz+nthreads-1)/nthreads);

    int nthreads1 = IRIS_CUDA_NTHREADS_Z;
    int nthreads2 = IRIS_CUDA_NTHREADS_Z;
    int nthreads3 = IRIS_CUDA_NTHREADS_Z;

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);
for(int vv=1;vv<10;vv++)
    kspace_eng_kernel<<<blocks,threads>>>(in_rho_phi, out_Ek_vir, nx, ny, nz, s2, compute_global_energy,compute_global_virial);
    HANDLE_LAST_CUDA_ERROR;
    cudaFree(in_rho_phi);
    cudaFree(out_Ek_vir);

    return 0;
}