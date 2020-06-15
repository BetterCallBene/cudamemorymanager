set(SOURCE_FILES
	include/cuda/common.hpp
	include/DeviceMemory.hpp
	include/HostMemory.hpp
	include/MemoryContainer.hpp
	include/MemoryManager.hpp
	include/MemoryManagerHostDev.hpp
	
	include/detail/MemoryContainer.inl.hpp
	include/helper/device_tuple.hpp
	include/helper/helper_c++14.hpp
	include/interface/IAllocator.hpp
	include/interface/IPointer.hpp
	
	source/DeviceMemory.cpp
	source/HostMemory.cpp
	source/MemoryManagerHostDev.cpp
	source/cuda/DeviceMemory.cu
	source/cuda/HostMemory.cu
)


add_library(MemoryManager
	STATIC
	${SOURCE_FILES}
)

add_library(${PROJECT_PREFIX}::${MODULE_PREFIX}::MemoryManager ALIAS MemoryManager)

target_include_directories(MemoryManager
	PRIVATE
		source
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	)

target_link_libraries(MemoryManager
	PUBLIC
        ${CUDA_LIBRARIES}
		TP_rte_api
)

set_target_properties(MemoryManager PROPERTIES
	FOLDER libraries
	OUTPUT_NAME ${PROJECT_PREFIX}-lib-trapla-memorymanager
	)

install(DIRECTORY include/
	DESTINATION include
	)

install(TARGETS MemoryManager
	EXPORT Targets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)
