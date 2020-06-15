#ifndef TRAPLA_MEMORYMANAGER_IPOINTER_HPP_
#define TRAPLA_MEMORYMANAGER_IPOINTER_HPP_

#include <Rte_Type.h>

namespace TraPla
{
namespace future
{
class IPointer
{
public:
    IPointer() = default;
    virtual ~IPointer() = default;
    virtual uint8* getPointer() const = 0;
    virtual void setPointer(uint8* ptr) = 0;
    virtual uint64 getSize() const = 0;
};
} // namespace future
} // namespace TraPla
#endif // TRAPLA_MEMORYMANAGER_IPOINTER_HPP_
