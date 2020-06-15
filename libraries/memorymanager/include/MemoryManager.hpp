#ifndef MEMORYMANAGER_MEMORY_MANAGER_HPP_
#define MEMORYMANAGER_MEMORY_MANAGER_HPP_

#include <map>
#include <Rte_Type.h>
#include "interface/IAllocator.hpp"
#include "MemoryManagerHostDev.hpp"
#include "HostMemory.hpp"
#include "DeviceMemory.hpp"

namespace TraPla
{
namespace future
{
template<typename T_MemoryStructure>
class MemoryManager : public MemoryManagerHostDev<T_MemoryStructure>
{
public:
    MemoryManager() = default;
    virtual ~MemoryManager() = default;

    template<typename T_Container>
    T_Container* installOnHost()
    {
#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
        return install<T_Container>(this->template getHostMemoryStructure(), &_memoryElementMapHost);
#else
        return install<T_Container>(this->getHostMemoryStructure(), &_memoryElementMapHost);
#endif
    }
    template<typename T_Container>
    T_Container* installOnDevice()
    {
#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
        return install<T_Container>(this->template getDeviceMemoryStructure(), &_memoryElementMapDevice);
#else
        return install<T_Container>(this->getDeviceMemoryStructure(), &_memoryElementMapDevice);
#endif
    }

    void init();
    void cleanUp();

protected:
    using MemoryElementMap = std::map<uint8, IPointer*>;

    struct ContainerInfo
    {
        uint8 containerId = 0;
        uint64 containerSize = 0;
    };

    template<typename T_Container>
    const static ContainerInfo assignContainer(T_MemoryStructure&, T_Container**);

    template<typename T_Container>
    T_Container* install(T_MemoryStructure& memory, MemoryElementMap* elementMap)
    {
        T_Container* pContainer = nullptr;
        auto containerInfo = assignContainer<T_Container>(memory, &pContainer);

        if (isElementInstalled(*elementMap, containerInfo.containerId))
        {
            cleanUp();
            exit(EXIT_FAILURE);
        }
        pContainer->setQuantity(containerInfo.containerSize);
        elementMap->operator[](containerInfo.containerId) = pContainer;

        return pContainer;
    }

    void allocate();
    void deallocate();
    void assignMemoryPosition();

    static void helperAllocate(IAllocator* allocator, const MemoryElementMap& mapElement);
    static void helperDeallocate(IAllocator* allocator, const MemoryElementMap& mapElement);
    static void helperAssignMemoryPosition(IAllocator* memoryPtr, const MemoryElementMap& mapElement);
    static uint64 getSizeOfContainers(const MemoryElementMap& mapElement);
    static void cleanUpStage2(MemoryElementMap* elementMap);
    static boolean isElementInstalled(const MemoryElementMap& memory, const uint8 elementType);

    MemoryElementMap _memoryElementMapHost{};
    MemoryElementMap _memoryElementMapDevice{};

    HostMemory _hostMemory{};
    DeviceMemory _deviceMemory{};
};
} // namespace future
} // namespace TraPla
#include "detail/MemoryManager.inl.hpp"
#endif
