#ifndef MEMORYMANAGER_MEMORY_MANAGER_INL_HPP_
#define MEMORYMANAGER_MEMORY_MANAGER_INL_HPP_

#include <cstring>

using namespace TraPla::future;

template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::cleanUp()
{
    deallocate();
    _memoryElementMapHost.clear();
    _memoryElementMapDevice.clear();
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::init()
{
    allocate();
    assignMemoryPosition();
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::allocate()
{
    helperAllocate(&_hostMemory, _memoryElementMapHost);
    helperAllocate(&_deviceMemory, _memoryElementMapDevice);
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::deallocate()
{
    helperDeallocate(&_hostMemory, _memoryElementMapHost);
    helperDeallocate(&_deviceMemory, _memoryElementMapDevice);
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::assignMemoryPosition()
{
    helperAssignMemoryPosition(&_hostMemory, _memoryElementMapHost);
    helperAssignMemoryPosition(&_deviceMemory, _memoryElementMapDevice);
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::helperAllocate(IAllocator* allocator, const MemoryElementMap& mapElement)
{
    auto containerSize = getSizeOfContainers(mapElement);
    if (containerSize > 0)
    {
        allocator->allocate(containerSize);
    }
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::helperDeallocate(IAllocator* allocator, const MemoryElementMap& mapElement)
{
    auto containerSize = getSizeOfContainers(mapElement);
    if (containerSize > 0)
    {
        allocator->deallocate();
    }
}
template<typename T_MemoryStructure>
void MemoryManager<T_MemoryStructure>::helperAssignMemoryPosition(
    IAllocator* memoryPtr,
    const MemoryElementMap& mapElement)
{
    uint8* currentPos = memoryPtr->getPointer();

    for (auto it = mapElement.begin(); it != mapElement.end(); ++it)
    {
        IPointer* element = it->second;
        element->setPointer(currentPos);
        currentPos += element->getSize();
    }
}
template<typename T_MemoryStructure>
uint64 MemoryManager<T_MemoryStructure>::getSizeOfContainers(const MemoryElementMap& mapElement)
{
    uint64 sum_size = 0;
    for (auto it = mapElement.begin(); it != mapElement.end(); ++it)
    {
        IPointer* element = it->second;
        sum_size += element->getSize();
    }
    return sum_size;
}
template<typename T_MemoryStructure>
boolean MemoryManager<T_MemoryStructure>::isElementInstalled(const MemoryElementMap& memory, const uint8 elementType)
{
    boolean value = false;

    if (memory.find(elementType) != memory.end())
    {
        value = true;
    }

    return value;
}

#endif // MEMORYMANAGER_MEMORY_MANAGER_INL_HPP_