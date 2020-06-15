//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \file          libraries/common/include/cuda/common.hpp
/// \brief         Provide defines and common functions for device functions.
/// \details       Combines a vehicle state with a cost value and a transition.
/// \responsible   Christian Reinl (christian.reinl@audi.de)
/// \module        TrajectoryPlanning
/// \author        EFS - Elektronische Fahrwerksysteme GmbH
/// \project       Automated Parking Software Development
/// \copyright     (c) 2019 Volkswagen AG (EFFP/1). All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef LIBRARIES_DYNAMICPLANNER_INCLUDE_CUDA_COMMON_HPP_
#define LIBRARIES_DYNAMICPLANNER_INCLUDE_CUDA_COMMON_HPP_
#include <iostream>

#ifdef __CUDACC__
#    include <cuda_runtime_api.h>
#    include <device_atomic_functions.h>
#    define CUDA_HOST __host__
#    define CUDA_DEV __device__
#    define CUDA_DEV_FORCE_INLINE __device__ __forceinline__
#    define CUDA_HOST_FORCE_INLINE __host__ __forceinline__
#    define CUDA_HOSTDEV __host__ __device__
#    define CUDA_HOSTDEV_FORCE_INLINE __host__ __device__ __forceinline__
#    define CUDA_GLOBAL __global__
#    define CUDA_GLOBAL_FORCE_INLINE __global__ __forceinline__
#else
#    define CUDA_HOST
#    define CUDA_DEV
#    define CUDA_DEV_FORCE_INLINE inline
#    define CUDA_HOST_FORCE_INLINE inline
#    define CUDA_HOSTDEV
#    define CUDA_HOSTDEV_FORCE_INLINE inline
#    define CUDA_GLOBAL
#    define CUDA_GLOBAL_FORCE_INLINE inline
#endif

namespace TraPla
{
namespace gpu
{
#ifdef __CUDACC__
CUDA_HOST_FORCE_INLINE void checkCudaError(cudaError_t err, const char* file, const int line, const char* func)
{
    if (cudaSuccess != err)
    {
        std::cout << "Cuda TraPla API Error: \n" << cudaGetErrorString(err);
    }
}
#    define CUDA_SAFE_CALL(expr) TraPla::gpu::checkCudaError((expr), __FILE__, __LINE__, __FUNCTION__)
#else
#    define CUDA_SAFE_CALL(expr) expr
#endif
} // namespace gpu
} // namespace TraPla
#endif // LIBRARIES_DYNAMICPLANNER_INCLUDE_CUDA_COMMON_HPP_
