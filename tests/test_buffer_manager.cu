#include <iostream>
#include "real.h"

#include "buffer_manager_gpu.h"

using namespace ORG_NCSA_IRIS;

int main()
{

    buffer_manager_gpu<iris_real> bm(3,2*1000*1000);
    try {
    iris_real* b1 = bm.take_buffer(1000*1000);
    iris_real* b2 = bm.take_buffer(1000);
    iris_real* b3 = bm.take_buffer(1000);
    
    std::cout<<cudaMemcpy(b2,b3,1000,cudaMemcpyDeviceToDevice)<<std::endl;
    bm.release_buffer(b3);
    iris_real* b4 = bm.take_buffer(1024);
    bm.release_buffer(b1);
    iris_real* b5 = bm.take_buffer(1024);
    } 
    catch (char const* msg)
    {
        std::cout<<msg<<std::endl;
    }
    return 0;
}

