#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Rte_Type.h>
#include "cuda/common.hpp"
#include "interface/IAllocator.hpp"
#include "HostMemory.hpp"
#include "DeviceMemory.hpp"
#include "interface/IPointer.hpp"
#include "MemoryManager.hpp"
#include "interface/IMemoryContainer.hpp"

namespace TraPla
{
namespace future
{
enum class MemoryElementType : uint8
{
    Dummy1Container = 0,
    Dummy2Container,
    Dummy3Container
};
class Dummy1Container : public IMemoryContainer
{
public:
    Dummy1Container() = default;
    virtual ~Dummy1Container() = default;

    uint8* getPointer() const override
    {
        return _ptr;
    }
    void setPointer(uint8* ptr) override
    {
        _ptr = ptr;
    }
    uint64 getSize() const override
    {
        return 1;
    }

    uint64 getQuantity() const override
    {
        return 1;
    }

    void setQuantity(uint64 quantity) override
    {
    }

private:
    uint8* _ptr;
};

class Dummy2Container : public IMemoryContainer
{
public:
    Dummy2Container() = default;
    virtual ~Dummy2Container() = default;

    uint8* getPointer() const override
    {
        return _ptr;
    }
    void setPointer(uint8* ptr) override
    {
        _ptr = ptr;
    }
    uint64 getSize() const override
    {
        return 10;
    }

    uint64 getQuantity() const override
    {
        return 10;
    }

    void setQuantity(uint64 quantity) override
    {
    }

private:
    uint8* _ptr;
};

class Dummy3Container : public IMemoryContainer
{
public:
    Dummy3Container() = default;
    virtual ~Dummy3Container() = default;

    uint8* getPointer() const override
    {
        return _ptr;
    }
    void setPointer(uint8* ptr) override
    {
        _ptr = ptr;
    }
    uint64 getSize() const override
    {
        return 100;
    }

    uint64 getQuantity() const override
    {
        return 10;
    }

    void setQuantity(uint64 quantity) override
    {
    }

private:
    uint8* _ptr;
};
struct MemoryStructure
{
    Dummy1Container dummy1Container{};
    Dummy2Container dummy2Container{};
    Dummy3Container dummy3Container{};
};

template<>
template<>
const MemoryManager<MemoryStructure>::ContainerInfo MemoryManager<MemoryStructure>::assignContainer<Dummy1Container>(
    MemoryStructure& memory,
    Dummy1Container** container)
{
    *container = &memory.dummy1Container;
    ContainerInfo containerInfo;
    containerInfo.containerId = static_cast<uint8>(MemoryElementType::Dummy1Container);
    containerInfo.containerSize = 1;
    return containerInfo;
}
template<>
template<>
const MemoryManager<MemoryStructure>::ContainerInfo MemoryManager<MemoryStructure>::assignContainer<Dummy2Container>(
    MemoryStructure& memory,
    Dummy2Container** container)
{
    *container = &memory.dummy2Container;
    ContainerInfo containerInfo;
    containerInfo.containerId = static_cast<uint8>(MemoryElementType::Dummy2Container);
    containerInfo.containerSize = 10;
    return containerInfo;
}
template<>
template<>
const MemoryManager<MemoryStructure>::ContainerInfo MemoryManager<MemoryStructure>::assignContainer<Dummy3Container>(
    MemoryStructure& memory,
    Dummy3Container** container)
{
    *container = &memory.dummy3Container;
    ContainerInfo containerInfo;
    containerInfo.containerId = static_cast<uint8>(MemoryElementType::Dummy3Container);
    containerInfo.containerSize = 100;
    return containerInfo;
}

template<>
template<>
CUDA_HOSTDEV const Dummy1Container* MemoryManagerHostDev<MemoryStructure>::getContainer<Dummy1Container>()
{
    return &getPlatformMemory()->dummy1Container;
}
template<>
template<>
CUDA_HOSTDEV const Dummy2Container* MemoryManagerHostDev<MemoryStructure>::getContainer<Dummy2Container>()
{
    return &getPlatformMemory()->dummy2Container;
}
template<>
template<>
CUDA_HOSTDEV const Dummy3Container* MemoryManagerHostDev<MemoryStructure>::getContainer<Dummy3Container>()
{
    return &getPlatformMemory()->dummy3Container;
}
} // namespace future
} // namespace TraPla
class AccessTests : public testing::Test
{
public:
    AccessTests()
    {
    }

protected:
};

namespace TraPla
{
namespace future
{
class FriendMemoryManager : public MemoryManager<MemoryStructure>
{
public:
    FriendMemoryManager() = default;
    ~FriendMemoryManager() = default;

    FRIEND_TEST(AccessTests, installHostContainers);
    FRIEND_TEST(AccessTests, installDeviceContainers);
    FRIEND_TEST(AccessTests, calculateSizeOfContainers);
    FRIEND_TEST(AccessTests, allocateMemory);
    FRIEND_TEST(AccessTests, initializeMemoryManager);
};

TEST_F(AccessTests, allocateMemory)
{
    FriendMemoryManager memoryManager;

    auto* dummy1Container = memoryManager.installOnHost<Dummy1Container>();
    auto* dummy2Container = memoryManager.installOnHost<Dummy2Container>();
    auto* dummy3Container = memoryManager.installOnHost<Dummy3Container>();

    ASSERT_TRUE(dummy1Container != nullptr);
    memoryManager.allocate();
    ASSERT_TRUE(dummy1Container->getPointer() == nullptr);
    ASSERT_TRUE(dummy1Container->getPointer() != memoryManager._hostMemory.getPointer());
    memoryManager.cleanUp();
    SUCCEED();
}

TEST_F(AccessTests, initializeMemoryManager)
{
    FriendMemoryManager memoryManager;

    auto* dummy1Container = memoryManager.installOnHost<Dummy1Container>();
    auto* dummy2Container = memoryManager.installOnHost<Dummy2Container>();
    auto* dummy3Container = memoryManager.installOnHost<Dummy3Container>();

    ASSERT_TRUE(dummy1Container != nullptr);

    memoryManager.init();
    ASSERT_TRUE(dummy1Container->getPointer() != nullptr);
    ASSERT_TRUE(dummy1Container->getPointer() == memoryManager._hostMemory.getPointer());

    ASSERT_TRUE(dummy2Container->getPointer() != nullptr);
    ASSERT_TRUE(dummy2Container->getPointer() == memoryManager._hostMemory.getPointer() + dummy1Container->getSize());

    ASSERT_TRUE(dummy3Container->getPointer() != nullptr);
    ASSERT_TRUE(
        dummy3Container->getPointer()
        == memoryManager._hostMemory.getPointer() + dummy1Container->getSize() + dummy2Container->getSize());

    memoryManager.cleanUp();
    SUCCEED();
}

TEST_F(AccessTests, installHostContainers)
{
    FriendMemoryManager memoryManager;

    FriendMemoryManager::MemoryElementMap& memory = memoryManager._memoryElementMapHost;

    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    auto* dummy1Container = memoryManager.installOnHost<Dummy1Container>();

    ASSERT_TRUE(dummy1Container != nullptr);
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    auto* dummy2Container = memoryManager.installOnHost<Dummy2Container>();

    ASSERT_TRUE(dummy2Container != nullptr);
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    auto* progressRangeCalculator = memoryManager.installOnHost<Dummy3Container>();

    ASSERT_TRUE(progressRangeCalculator != nullptr);
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    memoryManager.cleanUp();

    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));
}

TEST_F(AccessTests, installDeviceContainers)
{
    FriendMemoryManager memoryManager;

    FriendMemoryManager::MemoryElementMap& memory = memoryManager._memoryElementMapDevice;

    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    auto* dummy1Container = memoryManager.installOnDevice<Dummy1Container>();
    ASSERT_TRUE(dummy1Container != nullptr);

    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    auto* dummy2Container = memoryManager.installOnDevice<Dummy2Container>();
    ASSERT_TRUE(dummy2Container != nullptr);

    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    auto* progressRangeCalculator = memoryManager.installOnDevice<Dummy3Container>();
    ASSERT_TRUE(progressRangeCalculator != nullptr);

    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_TRUE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));

    memoryManager.cleanUp();

    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy1Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy2Container)));
    ASSERT_FALSE(
        FriendMemoryManager::isElementInstalled(memory, static_cast<uint8>(MemoryElementType::Dummy3Container)));
}
// TEST_F(AccessTests, crash)
// {
//     MemoryManager memoryManager;

//     auto* dummy1Container = memoryManager.installOnDevice<Dummy1Container>();
//     auto* explorationContainer2 = memoryManager.installOnDevice<Dummy1Container>();

//     memoryManager.cleanUp();
// }

TEST_F(AccessTests, get_special_container)
{
    MemoryManager<MemoryStructure> memoryManager;

    auto* dummy1Container = memoryManager.installOnHost<Dummy1Container>();
    auto* explorationContainer2 = memoryManager.getContainer<Dummy1Container>();

    ASSERT_TRUE(dummy1Container != nullptr);
    ASSERT_TRUE(explorationContainer2 != nullptr);

    ASSERT_TRUE(dummy1Container == explorationContainer2);

    memoryManager.cleanUp();

    const Dummy1Container* nullExplorationState = memoryManager.getContainer<Dummy1Container>();

    //ASSERT_TRUE(nullExplorationState == nullptr);  Should be zero currently not working
    //ASSERT_TRUE(explorationContainer2 == nullptr); Should be zero currently not working
}

TEST_F(AccessTests, calculateSizeOfContainers)
{
    FriendMemoryManager memoryManager;

    auto* dummy1Container = memoryManager.installOnHost<Dummy1Container>();

    auto sizeExplorationContainer = memoryManager.getSizeOfContainers(memoryManager._memoryElementMapHost);
    ASSERT_EQ(sizeExplorationContainer, dummy1Container->getSize());

    memoryManager.cleanUp();

    auto sizeEmpty = memoryManager.getSizeOfContainers(memoryManager._memoryElementMapHost);
    ASSERT_EQ(sizeEmpty, 0);

    auto* dummy2Container = memoryManager.installOnHost<Dummy2Container>();
    auto* dummy3Container = memoryManager.installOnHost<Dummy3Container>();

    auto sizeMaskEntryPlusProgressRangeContainer =
        memoryManager.getSizeOfContainers(memoryManager._memoryElementMapHost);
    ASSERT_TRUE(sizeMaskEntryPlusProgressRangeContainer == dummy2Container->getSize() + dummy3Container->getSize());

    memoryManager.cleanUp();

    auto sizeEmpty2 = memoryManager.getSizeOfContainers(memoryManager._memoryElementMapHost);
    ASSERT_EQ(sizeEmpty2, 0);
}
} // namespace future
} // namespace TraPla