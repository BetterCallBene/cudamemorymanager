#include "DeviceMemory.hpp"
#include "cuda/common.hpp"

void TraPla::future::DeviceMemory::allocate(uint64 size)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&_device_ptr, size));
}

void TraPla::future::DeviceMemory::deallocate()
{
    CUDA_SAFE_CALL(cudaFree(_device_ptr));
}