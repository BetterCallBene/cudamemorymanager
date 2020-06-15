#ifndef TRAPLA_MEMORYMANAGER_MEMORYMANAGER_HOSTDEV_HPP_
#define TRAPLA_MEMORYMANAGER_MEMORYMANAGER_HOSTDEV_HPP_
#include <Rte_Type.h>
#include "cuda/common.hpp"

namespace TraPla
{
namespace future
{
template<typename T_MemoryStructure>
class MemoryManagerHostDev
{
public:
    MemoryManagerHostDev() = default;
    virtual ~MemoryManagerHostDev() = default;

    template<typename T_Container>
    CUDA_HOSTDEV const T_Container* getContainer();

protected:
    T_MemoryStructure& getDeviceMemoryStructure()
    {
        return _deviceMemoryStructure;
    }

    T_MemoryStructure& getHostMemoryStructure()
    {
        return _hostMemoryStructure;
    }

    CUDA_HOSTDEV
    const T_MemoryStructure* getPlatformMemory()
    {
#ifndef __CUDA_ARCH__
        return &_hostMemoryStructure;
#else
        return &_deviceMemoryStructure;
#endif
    }

protected:
    T_MemoryStructure _deviceMemoryStructure{};
    T_MemoryStructure _hostMemoryStructure{};
};
} // namespace future
} // namespace TraPla

#endif // TRAPLA_MEMORYMANAGER_MEMORYMANAGER_HOSTDEV_HPP_