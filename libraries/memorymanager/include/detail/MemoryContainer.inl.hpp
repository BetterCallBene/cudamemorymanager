#ifndef TRAPLA_MEMORYMANAGER_MEMORY_CONTAINER_INL_HPP_
#define TRAPLA_MEMORYMANAGER_MEMORY_CONTAINER_INL_HPP_

template<template<typename...> class T_Item, typename... Types>
TraPla::future::MemoryContainer<T_Item<Types...>>::MemoryContainer(const uint64 quantity)
    : _quantity(quantity)
{
}

template<template<typename...> class T_Item, typename... Types>
uint8* TraPla::future::MemoryContainer<T_Item<Types...>>::getPointer() const
{
    return _ptr;
}

template<template<typename...> class T_Item, typename... Types>
uint64 TraPla::future::MemoryContainer<T_Item<Types...>>::getSize() const
{
    return _elementSize * _quantity;
}

template<template<typename...> class T_Item, typename... Types>
void TraPla::future::MemoryContainer<T_Item<Types...>>::setPointer(uint8* ptr)
{
    _ptr = ptr;
}

template<template<typename...> class T_Item, typename... Types>
void TraPla::future::MemoryContainer<T_Item<Types...>>::setQuantity(uint64 quantity)
{
    _quantity = quantity;
}

template<template<typename...> class T_Item, typename... Types>
CUDA_HOSTDEV uint64 TraPla::future::MemoryContainer<T_Item<Types...>>::getQuantity() const
{
    return _quantity;
}

template<template<typename...> class T_Item, typename... Types>
CUDA_HOSTDEV typename TraPla::future::MemoryContainer<T_Item<Types...>>::refT TraPla::future::MemoryContainer<
    T_Item<Types...>>::getElementAt(uint64 index) const
{
    return get(_ptr, index, _quantity, std::make_integer_sequence<unsigned, sizeof...(Types)>());
}

#endif // TRAPLA_MEMORYMANAGER_MEMORY_CONTAINER_INL_HPP_