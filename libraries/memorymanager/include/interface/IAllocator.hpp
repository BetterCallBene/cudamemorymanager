#ifndef TRAPLA_MEMORYMANAGER_IALLOCATOR_HPP_
#define TRAPLA_MEMORYMANAGER_IALLOCATOR_HPP_
#include <Rte_Type.h>

namespace TraPla
{
namespace future
{
class IAllocator
{
public:
    IAllocator() = default;
    virtual ~IAllocator() = default;

    virtual void allocate(uint64 size) = 0;
    virtual void deallocate() = 0;

    virtual uint8* getPointer() const = 0;
};
} // namespace future
} // namespace TraPla

#endif