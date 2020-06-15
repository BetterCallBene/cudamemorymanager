#ifndef TRAPLA_MEMORYMANAGER_MEMORY_CONTAINER_HPP_
#define TRAPLA_MEMORYMANAGER_MEMORY_CONTAINER_HPP_

#include <Rte_Type.h>
#include "cuda/common.hpp"
#include "interface/IMemoryContainer.hpp"
#include "helper/device_tuple.hpp"
#include "helper/helper_c++14.hpp"

namespace TraPla
{
namespace future
{
template<typename T_Item>
class MemoryContainer;

template<template<typename...> class T_Item, typename... Types>
class MemoryContainer<T_Item<Types...>> : public IMemoryContainer
{
public:
    //ToDo: using refT = T&;
    using refT = T_Item<Types...>;

    CUDA_HOSTDEV
    MemoryContainer(){};
    CUDA_HOSTDEV
    MemoryContainer(const uint64 quantity);
    virtual ~MemoryContainer() = default;

    uint8* getPointer() const override;
    uint64 getSize() const override;
    void setPointer(uint8* ptr) override;
    void setQuantity(uint64 quantity) override;
    CUDA_HOSTDEV
    uint64 getQuantity() const override;
    CUDA_HOSTDEV
    refT getElementAt(uint64 index) const;

protected:
    /***thread idx      0   1       2   3   4..              VariableIndex                        Type ***/ // Length = quantity
    /**Pointer pos.:    ptr ptr + indx
    /***x                                                     0              float **/  // Length of row: quantity * sizeof(Types)
    /** y                                                     1              float     **/
    /** psi                                                   2              float      **/
    /** cost                                                  3              int    **/
    // http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/docs/online/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html
    template<unsigned... VariableIndex>
    CUDA_HOSTDEV constexpr static refT get(
        uint8* ptr,
        const uint64 threadIndex,
        const uint64 quantity,
        std::integer_sequence<unsigned, VariableIndex...>)
    {
        return refT{
            (reinterpret_cast<Types>(ptr + CalcIndexPosition<VariableIndex, typename refT::basetype>::value(quantity))
             + threadIndex)...};
    }

    // Die Solution: https://godbolt.org/z/-PQwmd
protected:
    /* data */
    uint8* _ptr = nullptr;
    static constexpr uint64 _elementSize = refT::getSize();
    uint64 _quantity = 0;
};
} // namespace future
} // namespace TraPla
#include "detail/MemoryContainer.inl.hpp"
#endif // TRAPLA_MEMORYMANAGER_MEMORY_CONTAINER_HPP_