#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <unistd.h>
#include <stdexcept>

#define HANDLE_ERROR(res)						\
    if(res != cudaSuccess) {						\
	printf("CUDA Error: %s - %s\n", cudaGetErrorName(res), cudaGetErrorString(res)); \
	throw std::runtime_error("CUDA Exception occured");		\
    }

#define N 1000

struct bahor {
       float t[4];
};
    
int main()
{
    int *x;
    cudaMallocHost(&x, 1000);
    memset(x, 0, 1000*sizeof(int));
    cudaFree(x);
    
    srand(time(NULL));
    
    int *keys_cpu = (int *)malloc(N*sizeof(int));
    bahor *values_cpu = (bahor *)malloc(N*sizeof(bahor));
    
    for(int i=0;i<N;i++) {
	int v = rand() % N;
	keys_cpu[i] = v;
	values_cpu[i].t[0] = v * 2.0;
	values_cpu[i].t[1] = v * 4.0;
	values_cpu[i].t[2] = v * 8.0;
	values_cpu[i].t[3] = v * 16.0;
    }

    int *keys_gpu;
    bahor *values_gpu;

    cudaError_t res;
    res = cudaMalloc(&keys_gpu, N*sizeof(int)); HANDLE_ERROR(res);
    res = cudaMalloc(&values_gpu, N*sizeof(bahor));  HANDLE_ERROR(res);
    res = cudaMemcpy(keys_gpu, keys_cpu, N*sizeof(int), cudaMemcpyDefault);  HANDLE_ERROR(res);
    res = cudaMemcpy(values_gpu, values_cpu, N*sizeof(bahor), cudaMemcpyDefault);  HANDLE_ERROR(res);

    thrust::device_ptr<int> keys_ptr(keys_gpu);
    thrust::device_ptr<bahor> values_ptr(values_gpu);
    
    thrust::sort_by_key(keys_ptr, keys_ptr+N, values_ptr);

    res = cudaMemcpy(keys_cpu, keys_gpu, N*sizeof(int), cudaMemcpyDefault);  HANDLE_ERROR(res);
    res = cudaMemcpy(values_cpu, values_gpu, N*sizeof(bahor), cudaMemcpyDefault);  HANDLE_ERROR(res);
    

    for(int i=0;i<N;i++) {
	printf("K = %d, V = (%f, %f, %f, %f)\n", keys_cpu[i], values_cpu[i].t[0], values_cpu[i].t[1], values_cpu[i].t[2], values_cpu[i].t[3]);
    }

    res = cudaFree(keys_gpu);  HANDLE_ERROR(res);
    res = cudaFree(values_gpu);  HANDLE_ERROR(res);
    
    free(keys_cpu);
    free(values_cpu);
}
