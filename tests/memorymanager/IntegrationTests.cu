#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <array>
#include <Rte_Type.h>
#include "helper/device_tuple.hpp"
#include "helper/helper_c++14.hpp"
#include "MemoryContainer.hpp"
#include "MemoryManager.hpp"
#include "cuda/CudaUnitTest.hpp"

constexpr uint8 firstContainerId = 0u;
constexpr uint8 secondContainerId = 1u;
constexpr uint64 firstContainerSize = 10u;
constexpr uint64 secondContainerSize = 5u;

namespace TraPla
{
namespace future
{
template<typename... args_t>
struct DummyType5 : public tuple<args_t...>
{
    using tuple<args_t...>::tuple;
    using basetype = tuple<args_t...>;

    CUDA_HOSTDEV DummyType5(const args_t&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyType5(args_t&&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyType5(const basetype& rhs)
        : basetype(rhs)
    {
    }
    CUDA_HOSTDEV
    static constexpr uint64 getSize()
    {
        return CalcElementsSize<args_t...>::value;
    }

    CUDA_HOSTDEV
    auto& b()
    {
        return gpu::get<0>(*this);
    }
    CUDA_HOSTDEV
    auto& x()
    {
        return gpu::get<1>(*this);
    }
    CUDA_HOSTDEV
    auto& y()
    {
        return gpu::get<2>(*this);
    }
    CUDA_HOSTDEV
    auto& z()
    {
        return gpu::get<3>(*this);
    }

    CUDA_HOSTDEV
    auto& id()
    {
        return gpu::get<4>(*this);
    }
};

template<typename... args_t>
struct DummyType2 : public tuple<args_t...>
{
    using tuple<args_t...>::tuple;
    using basetype = tuple<args_t...>;

    CUDA_HOSTDEV DummyType2(const args_t&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyType2(args_t&&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyType2(const basetype& rhs)
        : basetype(rhs)
    {
    }
    CUDA_HOSTDEV
    static constexpr uint64 getSize()
    {
        return CalcElementsSize<args_t...>::value;
    }

    CUDA_HOSTDEV
    auto& x()
    {
        return gpu::get<0>(*this);
    }
    CUDA_HOSTDEV
    auto& y()
    {
        return gpu::get<1>(*this);
    }
};

} // namespace future
} // namespace TraPla

class ProfilingTests : public testing::Test
{
public:
    ProfilingTests()
    {
    }

protected:
};

namespace TraPla
{
namespace future
{
using DummyDefinition5Value = DummyType5<float32*, float32*, float32*, float32*, float32*>;
using DummyDefinition2Value = DummyType2<float32*, float32*>;
using DummyMemoryContainer5Value = MemoryContainer<DummyDefinition5Value>;
using DummyMemoryContainer2Value = MemoryContainer<DummyDefinition2Value>;

struct DummyMemoryStructure
{
    DummyMemoryContainer5Value dummy1Container;
    DummyMemoryContainer2Value dummy2Container;
};

template<>
template<>
const MemoryManager<DummyMemoryStructure>::ContainerInfo MemoryManager<DummyMemoryStructure>::assignContainer<
    DummyMemoryContainer5Value>(DummyMemoryStructure& memory, DummyMemoryContainer5Value** container)
{
    *container = &memory.dummy1Container;
    ContainerInfo containerInfo;
    containerInfo.containerId = firstContainerId;
    containerInfo.containerSize = firstContainerSize;
    return containerInfo;
}

template<>
template<>
const MemoryManager<DummyMemoryStructure>::ContainerInfo MemoryManager<DummyMemoryStructure>::assignContainer<
    DummyMemoryContainer2Value>(DummyMemoryStructure& memory, DummyMemoryContainer2Value** container)
{
    *container = &memory.dummy2Container;
    ContainerInfo containerInfo;
    containerInfo.containerId = secondContainerId;
    containerInfo.containerSize = secondContainerSize;
    return containerInfo;
}

template<>
template<>
CUDA_HOSTDEV const DummyMemoryContainer5Value* MemoryManagerHostDev<DummyMemoryStructure>::getContainer<
    DummyMemoryContainer5Value>()
{
    return &getPlatformMemory()->dummy1Container;
}
template<>
template<>
CUDA_HOSTDEV const DummyMemoryContainer2Value* MemoryManagerHostDev<DummyMemoryStructure>::getContainer<
    DummyMemoryContainer2Value>()
{
    return &getPlatformMemory()->dummy2Container;
}

using DummyManager = MemoryManager<DummyMemoryStructure>;
using DummyManagerBasic = MemoryManagerHostDev<DummyMemoryStructure>;
template<class Iter>
void fillWithRandomIntValues(Iter start, Iter end, float min, float max)
{
    static std::random_device rd; // you only need to initialize it once
    static std::mt19937 mte(rd()); // this is a relative big object to create

    std::uniform_real_distribution<float32> dist(min, max);

    std::generate(start, end, [&]() { return dist(mte); });
}

struct helperUpload
{
    DummyManagerBasic manager{};
    float32* random_values{};
};

constexpr uint64 countOfRandomSize = 5 * firstContainerSize + 2 * secondContainerSize;
constexpr uint64 randomSize = countOfRandomSize * sizeof(float32);

GPU_DEFINE_TEST_BODY_UPLOAD(integration_test_memory_container, helperUpload, dataUpload)
{
    auto container5Values = dataUpload.manager.getContainer<DummyMemoryContainer5Value>();

    uint64 j = 0;
    for (uint64 i = 0u; i < container5Values->getQuantity(); i++)
    {
        auto element = container5Values->getElementAt(i);

        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.b(), 1e-8);
        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.x(), 1e-8);
        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.y(), 1e-8);
        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.z(), 1e-8);
        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.id(), 1e-8);
    }

    auto container2Values = dataUpload.manager.getContainer<DummyMemoryContainer2Value>();

    for (uint64 i = 0u; i < container2Values->getQuantity(); i++)
    {
        auto element = container2Values->getElementAt(i);

        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.x(), 1e-8);
        GPU_ASSERT_NEAR(dataUpload.random_values[j++], *element.y(), 1e-8);
    }
}

TEST_F(ProfilingTests, integration_test_memory_container)
{
    std::array<float32, countOfRandomSize> random_values;

    fillWithRandomIntValues(random_values.begin(), random_values.end(), 0, 1);
    DeviceMemory memoryDeviceRandomValues;
    memoryDeviceRandomValues.allocate(randomSize);

    DummyManager manager;
    auto containerHost1 = manager.installOnHost<DummyMemoryContainer5Value>();
    auto containerHost2 = manager.installOnHost<DummyMemoryContainer2Value>();

    auto containerDevice1 = manager.installOnDevice<DummyMemoryContainer5Value>();
    auto containerDevice2 = manager.installOnDevice<DummyMemoryContainer2Value>();

    uint32 j = 0u;
    manager.init();
    for (uint64 i = 0u; i < containerHost1->getQuantity(); i++)
    {
        auto element = containerHost1->getElementAt(i);
        *element.b() = random_values[j++];
        *element.x() = random_values[j++];
        *element.y() = random_values[j++];
        *element.z() = random_values[j++];
        *element.id() = random_values[j++];
    }

    for (uint64 i = 0u; i < containerHost2->getQuantity(); i++)
    {
        auto element = containerHost2->getElementAt(i);
        *element.x() = random_values[j++];
        *element.y() = random_values[j++];
    }

    cudaMemcpy(
        containerDevice1->getPointer(),
        containerHost1->getPointer(),
        containerHost1->getSize(),
        cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaMemcpy(
        containerDevice2->getPointer(),
        containerHost2->getPointer(),
        containerHost2->getSize(),
        cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaMemcpy(
        memoryDeviceRandomValues.getPointer(),
        random_values.data(),
        randomSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice);

    helperUpload dataUpload{};
    dataUpload.manager = *dynamic_cast<DummyManagerBasic*>(&manager);
    dataUpload.random_values = (float32*)memoryDeviceRandomValues.getPointer(); //random_values.data();

    GPU_TEST_EXECUTE_UPLOAD(integration_test_memory_container, 1, 1, dataUpload);

    manager.cleanUp();
    SUCCEED();
}

} // namespace future
} // namespace TraPla
