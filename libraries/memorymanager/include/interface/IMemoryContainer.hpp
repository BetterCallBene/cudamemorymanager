#ifndef MEMORYMANAGER_IMEMORY_CONTAINER_HPP_
#define MEMORYMANAGER_IMEMORY_CONTAINER_HPP_

#include <Rte_Type.h>
#include "cuda/common.hpp"
#include "interface/IPointer.hpp"

namespace TraPla
{
namespace future
{
class IMemoryContainer : public IPointer
{
public:
    IMemoryContainer(/* args */) = default;
    virtual ~IMemoryContainer() = default;

    virtual void setQuantity(uint64 quantity) = 0;
    CUDA_HOSTDEV
    virtual uint64 getQuantity() const = 0;
};
} // namespace future
} // namespace TraPla
#endif // MEMORYMANAGER_IMEMORY_CONTAINER_HPP_