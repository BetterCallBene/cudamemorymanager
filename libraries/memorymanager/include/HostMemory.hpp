#ifndef TRAPLA_MEMORYMANAGER_HOSTMEMORY_HPP_
#define TRAPLA_MEMORYMANAGER_HOSTMEMORY_HPP_

#include <Rte_Type.h>
#include "interface/IAllocator.hpp"

namespace TraPla
{
namespace future
{
class HostMemory : public IAllocator
{
public:
    HostMemory() = default;
    ~HostMemory() = default;

    HostMemory(const HostMemory& other) = delete;
    HostMemory& operator=(const HostMemory& other) = delete;
    HostMemory(HostMemory&& other) = delete;

    void allocate(uint64 size) override;
    void deallocate() override;

    uint8* getPointer() const override;

private:
    uint8* _host_ptr = nullptr;
};
} // namespace future
} // namespace TraPla
#endif // TRAPLA_MEMORYMANAGER_HOSTMEMORY_HPP_