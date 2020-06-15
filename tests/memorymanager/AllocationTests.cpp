#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "HostMemory.hpp"
#include "DeviceMemory.hpp"

class AllocationTests : public testing::Test
{
public:
    AllocationTests()
    {
    }

protected:
};

namespace TraPla
{
namespace future
{
TEST_F(AllocationTests, initial_allocation_hostmemory)
{
    constexpr uint64 memorySize = 1024u;
    HostMemory memory;
    memory.allocate(memorySize);
    memory.deallocate();
    SUCCEED();
}

TEST_F(AllocationTests, initial_allocation_devicememory)
{
    constexpr uint64 memorySize = 1024u;
    DeviceMemory memory;
    memory.allocate(memorySize);
    memory.deallocate();
    SUCCEED();
}
} // namespace future
} // namespace TraPla
