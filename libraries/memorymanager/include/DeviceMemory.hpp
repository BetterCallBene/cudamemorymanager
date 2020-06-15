#ifndef TRAPLA_MEMORYMANAGER_DEVICE_MEMORY_HPP_
#define TRAPLA_MEMORYMANAGER_DEVICE_MEMORY_HPP_

#include <Rte_Type.h>
#include "interface/IAllocator.hpp"

namespace TraPla
{
namespace future
{
class DeviceMemory : public IAllocator
{
public:
    DeviceMemory() = default;
    ~DeviceMemory() = default;

    DeviceMemory(const DeviceMemory& other) = delete;
    DeviceMemory& operator=(const DeviceMemory& other) = delete;
    DeviceMemory(DeviceMemory&& other) = delete;

    void allocate(uint64 size) override;
    void deallocate() override;

    uint8* getPointer() const override;

private:
    uint8* _device_ptr = nullptr;
};
} // namespace future
} // namespace TraPla
#endif // TRAPLA_MEMORYMANAGER_DEVICE_MEMORY_HPP_