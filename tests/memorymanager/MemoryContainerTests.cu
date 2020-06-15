#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <array>
#include <Rte_Type.h>
#include "cuda/common.hpp"
#include "cuda/CudaUnitTest.hpp"
#include "helper/device_tuple.hpp"
#include "helper/helper_c++14.hpp"
#include "interface/IPointer.hpp"
#include "HostMemory.hpp"
#include "DeviceMemory.hpp"
#include "MemoryContainer.hpp"

namespace TraPla
{
namespace future
{
template<typename... args_t>
struct DummyType : public tuple<args_t...>
{
    using tuple<args_t...>::tuple;
    using basetype = tuple<args_t...>;

    CUDA_HOSTDEV DummyType(const args_t&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyType(args_t&&... args)
        : basetype(args...)
    {
    }
    CUDA_HOSTDEV DummyType(const basetype& rhs)
        : basetype(rhs)
    {
    }

    CUDA_HOSTDEV
    static constexpr uint64 getSize()
    {
        return CalcElementsSize<args_t...>::value;
    }
    CUDA_HOSTDEV
    auto& w()
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
};

template<typename T_Item>
class FakeMemoryContainer : public MemoryContainer<T_Item>
{
public:
    using MemoryContainer<T_Item>::MemoryContainer;

    CUDA_HOSTDEV
    void setPointerOnDevice(uint8* ptr)
    {
        MemoryContainer<T_Item>::_ptr = ptr;
    }
};
} // namespace future
} // namespace TraPla

class MemoryContainerTests : public testing::Test
{
};

namespace TraPla
{
namespace future
{
TEST_F(MemoryContainerTests, check_calculate_memory_size)
{
    using DummyDefinition = DummyType<boolean*, float32*, uint16*, int32*>;

    constexpr uint64 countColumn = 10u;
    constexpr uint64 columnSize = sizeof(boolean) + sizeof(float32) + sizeof(uint16) + sizeof(int32);
    constexpr uint64 matrixSize = columnSize * countColumn;
    MemoryContainer<DummyDefinition> memoryContainerDummyDefinition(countColumn);

    ASSERT_EQ(memoryContainerDummyDefinition.getSize(), matrixSize);
}

TEST_F(MemoryContainerTests, get_element)
{
    using DummyDefinition = DummyType<boolean*, float32*, uint16*, int32*>;

    constexpr uint64 countDummyTypes = 10u;
    constexpr uint64 columnSize = sizeof(boolean) + sizeof(float32) + sizeof(uint16) + sizeof(int32);
    constexpr uint64 matrixSize = columnSize * countDummyTypes;
    MemoryContainer<DummyDefinition> memoryContainerDummyDefinition(countDummyTypes);

    ASSERT_EQ(memoryContainerDummyDefinition.getSize(), matrixSize);

    uint8* ptr = (uint8*)malloc(matrixSize);
    memoryContainerDummyDefinition.setPointer(ptr);
    ASSERT_EQ(countDummyTypes, memoryContainerDummyDefinition.getQuantity());

    int32 Cost = 20;

    for (uint64 i = 0; i < memoryContainerDummyDefinition.getQuantity(); i++)
    {
        auto dummyDefintion = memoryContainerDummyDefinition.getElementAt(i);
        *dummyDefintion.w() = true;
        *dummyDefintion.x() = float32(i);
        *dummyDefintion.y() = uint16(memoryContainerDummyDefinition.getQuantity() - i);
        *dummyDefintion.z() = Cost;
    }

    for (uint64 i = 0; i < memoryContainerDummyDefinition.getQuantity(); i++)
    {
        auto dummyDefintion = memoryContainerDummyDefinition.getElementAt(i);

        ASSERT_EQ(*dummyDefintion.w(), true);
        ASSERT_NEAR(*dummyDefintion.x(), float32(i), 1e-8);
        ASSERT_EQ(*dummyDefintion.y(), uint16(memoryContainerDummyDefinition.getQuantity() - i));
        ASSERT_EQ(*dummyDefintion.z(), Cost);
    }
}

template<class Iter>
void fillWithRandomIntValues(Iter start, Iter end, float min, float max)
{
    static std::random_device rd; // you only need to initialize it once
    static std::mt19937 mte(rd()); // this is a relative big object to create

    std::uniform_real_distribution<float32> dist(min, max);

    std::generate(start, end, [&]() { return dist(mte); });
}

TEST_F(MemoryContainerTests, test_with_random_values)
{
    using DummyDefinition = DummyType<float32*, float32*, float32*, float32*>;

    constexpr uint64 countDummyTypes = 10u;
    constexpr uint8 columnsNumbers = 4;
    constexpr uint64 columnSize = sizeof(float32) + sizeof(float32) + sizeof(float32) + sizeof(float32);
    constexpr uint64 matrixSize = columnSize * countDummyTypes;
    MemoryContainer<DummyDefinition> memoryContainerDummyDefinition(countDummyTypes);

    ASSERT_EQ(memoryContainerDummyDefinition.getSize(), matrixSize);

    uint8* ptr = (uint8*)malloc(matrixSize);
    memoryContainerDummyDefinition.setPointer(ptr);
    ASSERT_EQ(countDummyTypes, memoryContainerDummyDefinition.getQuantity());

    std::array<float32, columnsNumbers * countDummyTypes> random_values;
    fillWithRandomIntValues(random_values.begin(), random_values.end(), 0, 1);

    uint32 j = 0;

    for (uint64 i = 0; i < memoryContainerDummyDefinition.getQuantity(); i++)
    {
        DummyDefinition dummyDefintion = memoryContainerDummyDefinition.getElementAt(i);
        *dummyDefintion.w() = random_values[j++];
        *dummyDefintion.x() = random_values[j++];
        *dummyDefintion.y() = random_values[j++];
        *dummyDefintion.z() = random_values[j++];
    }
    j = 0;
    for (uint64 i = 0; i < memoryContainerDummyDefinition.getQuantity(); i++)
    {
        DummyDefinition dummyDefintion = memoryContainerDummyDefinition.getElementAt(i);

        ASSERT_NEAR(*dummyDefintion.w(), random_values[j++], 1e-8);
        ASSERT_NEAR(*dummyDefintion.x(), random_values[j++], 1e-8);
        ASSERT_NEAR(*dummyDefintion.y(), random_values[j++], 1e-8);
        ASSERT_NEAR(*dummyDefintion.z(), random_values[j++], 1e-8);
    }
}

constexpr uint8 columnsNumbers = 4;
constexpr uint64 countDummyTypes = 10u;
constexpr uint64 columnSize = sizeof(float32) + sizeof(float32) + sizeof(float32) + sizeof(float32);
constexpr uint64 matrixSize = columnSize * countDummyTypes;
constexpr uint64 countOfRandomSize = columnsNumbers * countDummyTypes;
constexpr uint64 randomSize = sizeof(float32) * countOfRandomSize;

struct helperUpload
{
    uint8* ptrDummyDefinition;
    float32* random_values;
};

GPU_DEFINE_TEST_BODY_UPLOAD(test_get_element_on_device, helperUpload, uploadData)
{
    using DummyDefinition = DummyType<float32*, float32*, float32*, float32*>;

    FakeMemoryContainer<DummyDefinition> memoryContainerDummyDefinition(countDummyTypes);
    memoryContainerDummyDefinition.setPointerOnDevice(uploadData.ptrDummyDefinition);
    uint32 j = 0;
    for (uint64 i = 0; i < memoryContainerDummyDefinition.getQuantity(); i++)
    {
        DummyDefinition dummyDefintion = memoryContainerDummyDefinition.getElementAt(i);

        GPU_ASSERT_NEAR(uploadData.random_values[j++], *dummyDefintion.w(), 1e-8);
        GPU_ASSERT_NEAR(uploadData.random_values[j++], *dummyDefintion.x(), 1e-8);
        GPU_ASSERT_NEAR(uploadData.random_values[j++], *dummyDefintion.y(), 1e-8);
        GPU_ASSERT_NEAR(uploadData.random_values[j++], *dummyDefintion.z(), 1e-8);
    }
}

TEST_F(MemoryContainerTests, test_get_element_on_device)
{
    using DummyDefinition = DummyType<float32*, float32*, float32*, float32*>;

    MemoryContainer<DummyDefinition> memoryContainerDummyDefinition(countDummyTypes);

    ASSERT_EQ(memoryContainerDummyDefinition.getSize(), matrixSize);

    HostMemory memoryHostDummyDefinition;
    DeviceMemory memoryDeviceDummyDefinition;
    DeviceMemory memoryDeviceRandomValues;

    memoryHostDummyDefinition.allocate(matrixSize);
    memoryDeviceDummyDefinition.allocate(matrixSize);

    memoryDeviceRandomValues.allocate(randomSize);

    memoryContainerDummyDefinition.setPointer(memoryHostDummyDefinition.getPointer());

    std::array<float32, countOfRandomSize> random_values;
    fillWithRandomIntValues(random_values.begin(), random_values.end(), 0, 1);

    uint32 j = 0;

    for (uint64 i = 0; i < memoryContainerDummyDefinition.getQuantity(); i++)
    {
        DummyDefinition dummyDefintion = memoryContainerDummyDefinition.getElementAt(i);
        *dummyDefintion.w() = random_values[j++];
        *dummyDefintion.x() = random_values[j++];
        *dummyDefintion.y() = random_values[j++];
        *dummyDefintion.z() = random_values[j++];
    }

    cudaMemcpy(
        memoryDeviceDummyDefinition.getPointer(),
        memoryHostDummyDefinition.getPointer(),
        matrixSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(
        memoryDeviceRandomValues.getPointer(),
        random_values.data(),
        randomSize,
        cudaMemcpyKind::cudaMemcpyHostToDevice);

    helperUpload uploadData;
    uploadData.ptrDummyDefinition = memoryDeviceDummyDefinition.getPointer();
    uploadData.random_values = (float32*)memoryDeviceRandomValues.getPointer();

    GPU_TEST_EXECUTE_UPLOAD(test_get_element_on_device, 1, 1, uploadData);

    memoryHostDummyDefinition.deallocate();
    memoryDeviceDummyDefinition.deallocate();
    memoryDeviceRandomValues.deallocate();
}
} // namespace future
} // namespace TraPla
