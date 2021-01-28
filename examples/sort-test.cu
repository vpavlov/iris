#include "fmm.h"
#include "real.h"

int main()
{
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    //------------------------
    
    
    //------------------------
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Elapsed time = %f ms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
