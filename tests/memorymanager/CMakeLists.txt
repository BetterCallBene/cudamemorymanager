set (SOURCE_FILES
	cuda/CudaUnitTest.hpp
	gtest_main.cpp
	HelperTests.cu
	AllocationTests.cpp
	MemoryContainerTests.cu
	MemoryManagerTests.cpp
	IntegrationTests.cu
)
set (DEPENDENCIES_LIBRARIES
	MemoryManager
	gtest_main gtest gmock_main gmock
)

add_executable(
	MemoryManager_tests
	${SOURCE_FILES}
)

target_link_libraries(MemoryManager_tests
	PRIVATE
	 $<$<PLATFORM_ID:Linux>:pthread> 
	 ${DEPENDENCIES_LIBRARIES}
)

add_test(
		NAME
		MemoryManager_tests
		COMMAND
		$<TARGET_FILE:MemoryManager_tests>
)