#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Rte_Type.h>
#include "helper/device_tuple.hpp"
#include "helper/helper_c++14.hpp"
#include "cuda/CudaUnitTest.hpp"

constexpr float errorToleranceFloat = 1e-5f;

class HelperTests : public testing::Test
{
public:
    HelperTests()
    {
    }

protected:
};

namespace TraPla
{
namespace future
{
#ifdef _WIN32
using namespace std;
#else
using namespace gpu;
#endif

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
};

GPU_DEFINE_TEST_BODY(create_read_tuple_on_device, tuple)
{
    constexpr int32 random_dummy_integer_value = 1;
    constexpr float32 random_dummy_float_value = 1.f;
    tuple<int32, float32> dummy_tuple = make_tuple(random_dummy_integer_value, random_dummy_float_value);

    auto actual_result_first_element = get<0>(dummy_tuple);
    auto actual_result_second_element = get<1>(dummy_tuple);
    GPU_ASSERT_INT_EQ(actual_result_first_element, random_dummy_integer_value);
    GPU_ASSERT_NEAR(actual_result_second_element, random_dummy_float_value, errorToleranceFloat);
}

TEST_F(HelperTests, create_read_tuple_on_device){GPU_TEST_EXECUTE(create_read_tuple_on_device, 1, 1)}

GPU_DEFINE_TEST_BODY(create_read_struct_tuple_on_device, tuple)
{
    int32 random_dummy_integer_value = 1;
    float32 random_dummy_float_value = 1.f;

    DummyType<int32, float32> testAttributeConstructor{random_dummy_integer_value, random_dummy_float_value};
    DummyType<int32, float32> testCopyConstructor{make_tuple(random_dummy_integer_value, random_dummy_float_value)};
    DummyType<int32, float32> testMoveConstructor{std::move(testCopyConstructor)};

    auto firstAttributeOfAttributeConstructor = get<0>(testAttributeConstructor);
    auto secondAttributeOfAttributeConstructor = get<1>(testAttributeConstructor);

    auto firstAttributeOfCopyConstructor = get<0>(testCopyConstructor);
    auto secondAttributeOfCopyConstructor = get<1>(testCopyConstructor);

    auto firstAttributeOfMoveConstructor = get<0>(testMoveConstructor);
    auto secondAttributeOfMoveConstructor = get<1>(testMoveConstructor);

    GPU_ASSERT_INT_EQ(firstAttributeOfAttributeConstructor, firstAttributeOfCopyConstructor);
    GPU_ASSERT_INT_EQ(firstAttributeOfCopyConstructor, firstAttributeOfMoveConstructor);

    GPU_ASSERT_NEAR(secondAttributeOfAttributeConstructor, secondAttributeOfCopyConstructor, errorToleranceFloat);
    GPU_ASSERT_NEAR(secondAttributeOfCopyConstructor, secondAttributeOfMoveConstructor, errorToleranceFloat);
}

TEST_F(HelperTests, create_read_struct_tuple_on_device){GPU_TEST_EXECUTE(create_read_struct_tuple_on_device, 1, 1)}

GPU_DEFINE_TEST_BODY(test_template_typelist_on_device, typelist)
{
    using TestType = tuple<float32, int16, int32, boolean>;
    GPU_ASSERT_INT_EQ((uint32)DataElementSize<TestType>::value<0>(), (uint32)sizeof(float32));
    GPU_ASSERT_INT_EQ((uint32)DataElementSize<TestType>::value<1>(), (uint32)sizeof(int16));
    GPU_ASSERT_INT_EQ((uint32)DataElementSize<TestType>::value<2>(), (uint32)sizeof(int32));
    GPU_ASSERT_INT_EQ((uint32)DataElementSize<TestType>::value<3>(), (uint32)sizeof(boolean));
}

TEST_F(HelperTests, test_template_typelist_on_device){GPU_TEST_EXECUTE(test_template_typelist_on_device, 1, 1)}

TEST_F(HelperTests, test_template_typelist)
{
    using TestType = tuple<float32, int16, int32, boolean>;
    ASSERT_EQ(DataElementSize<TestType>::value<0>(), sizeof(float32));
    ASSERT_EQ(DataElementSize<TestType>::value<1>(), sizeof(int16));
    ASSERT_EQ(DataElementSize<TestType>::value<2>(), sizeof(int32));
    ASSERT_EQ(DataElementSize<TestType>::value<3>(), sizeof(boolean));
}

GPU_DEFINE_TEST_BODY(test_calc_element_Size, dummyType)
{
    using DummyDefinition = DummyType<boolean, float32, uint16, int32>;
    DummyDefinition sizeCalc(true, 1e-5f, 10, 155);

    GPU_ASSERT_INT_EQ(
        (uint32)sizeCalc.getSize(), (uint32)(sizeof(boolean) + sizeof(float32) + sizeof(int16) + sizeof(int32)));
}

TEST_F(HelperTests, test_calc_element_Size)
{
    GPU_TEST_EXECUTE(test_calc_element_Size, 1, 1);
}

using DummyDefinition = DummyType<bool*, float*, short*, float*>;
using DummyDefinitionRet = DummyType<int32, int32, int32, int32>;

template<unsigned... VariableIndex>
CUDA_HOSTDEV static DummyDefinitionRet test_get_calc_index_pos(
    std::size_t threadIndex,
    std::integer_sequence<unsigned, VariableIndex...>)
{
    return DummyDefinitionRet{
        ((CalcIndexPosition<VariableIndex, typename DummyDefinition::basetype>::value(1)) + threadIndex)...};
}

GPU_DEFINE_TEST_BODY(test_calc_index_pos, dummyType)
{
    auto x = test_get_calc_index_pos(0, std::make_integer_sequence<unsigned, 4>{});
    auto pos_0 = get<0>(x);
    auto pos_1 = get<1>(x);
    auto pos_2 = get<2>(x);
    auto pos_3 = get<3>(x);
    GPU_ASSERT_INT_EQ(pos_0, 0);
    GPU_ASSERT_INT_EQ(pos_1, (uint32)(sizeof(bool)));
    GPU_ASSERT_INT_EQ(pos_2, (uint32)(sizeof(bool) + sizeof(float)));
    GPU_ASSERT_INT_EQ(pos_3, (uint32)(sizeof(bool) + sizeof(float) + sizeof(short)));
    GPU_ASSERT_INT_EQ(
        (uint32)DummyDefinition::getSize(), (uint32)(sizeof(bool) + sizeof(float) + sizeof(short) + sizeof(float)));
}

TEST_F(HelperTests, test_calc_index_pos)
{
    GPU_TEST_EXECUTE(test_calc_index_pos, 1, 1);
}

} // namespace future
} // namespace TraPla
