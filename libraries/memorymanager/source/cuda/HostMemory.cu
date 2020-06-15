#include "cuda/common.hpp"
#include "HostMemory.hpp"

void TraPla::future::HostMemory::allocate(uint64 size)
{
    CUDA_SAFE_CALL(cudaMallocHost((void**)&_host_ptr, size));  
}

void TraPla::future::HostMemory::deallocate()
{
    CUDA_SAFE_CALL(cudaFreeHost(_host_ptr));
}