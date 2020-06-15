#ifndef TRAPLA_CUDAUNITTEST_HPP
#define TRAPLA_CUDAUNITTEST_HPP

#include "stdio.h"
#include "cuda/common.hpp"

#define GPU_MOCK_METHOD_VOID_0(mock_function) \
public: \
    int count_##mock_function = 0; \
    __device__ void mock_function() \
    { \
        ++count_##mock_function; \
        return; \
    }

#define GPU_MOCK_METHOD_0(mock_function, out_type, return_value) \
public: \
    int count_##mock_function = 0; \
    __device__ out_type mock_function() \
    { \
        ++count_##mock_function; \
        return return_value; \
    }

#define GPU_MOCK_METHOD_VOID_1(mock_function, in_type) \
public: \
    int count_##mock_function = 0; \
    __device__ void mock_function(in_type) \
    { \
        ++count_##mock_function; \
        return; \
    }

#define GPU_MOCK_METHOD_1(mock_function, in_type, out_type, return_value) \
public: \
    int count_##mock_function = 0; \
    __host__ __device__ out_type mock_function(in_type) \
    { \
        ++count_##mock_function; \
        return return_value; \
    }

#define GPU_MOCK_METHOD_VOID_2(mock_function, in_type_1, in_type_2) \
public: \
    int count_##mock_function = 0; \
    __device__ void mock_function(in_type_1, in_type_2) \
    { \
        ++count_##mock_function; \
        return; \
    }

#define GPU_MOCK_METHOD_2(mock_function, in_type_1, in_type_2, out_type, return_value) \
public: \
    int count_##mock_function = 0; \
    __host__ __device__ out_type mock_function(in_type_1, in_type_2) \
    { \
        ++count_##mock_function; \
        return return_value; \
    }

#define GPU_MOCK_METHOD_VOID_3(mock_function, in_type_1, in_type_2, in_type_3) \
public: \
    int count_##mock_function = 0; \
    __device__ void mock_function(in_type_1, in_type_2, in_type_3) \
    { \
        ++count_##mock_function; \
        return; \
    }

#define GPU_MOCK_METHOD_3(mock_function, in_type_1, in_type_2, in_type_3, out_type, return_value) \
public: \
    int count_##mock_function = 0; \
    __device__ out_type mock_function(in_type_1, in_type_2, in_type_3) \
    { \
        ++count_##mock_function; \
        return return_value; \
    }

#define GPU_MOCK_METHOD_VOID_4(mock_function, in_type_1, in_type_2, in_type_3, in_type_4) \
public: \
    int count_##mock_function = 0; \
    __device__ void mock_function(in_type_1, in_type_2, in_type_3, in_type_4) \
    { \
        ++count_##mock_function; \
        return; \
    }

#define GPU_MOCK_METHOD_4(mock_function, in_type_1, in_type_2, in_type_3, in_type_4, out_type, return_value) \
public: \
    int count_##mock_function = 0; \
    __device__ out_type mock_function(in_type_1, in_type_2, in_type_3, in_type_4) \
    { \
        ++count_##mock_function; \
        return return_value; \
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Macros for test for functions that require one argument having a return value
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GPU_CALL_CHECK(mock_function, number_of_expected_calls, test_object) \
    if (number_of_expected_calls >= 0 \
        && test_object.get_##mock_function##_counter_from_mock() != number_of_expected_calls) \
    { \
        printf( \
            "Error detected on the GPU in file %s, line %d: Function mock_function was called %d times, expected %d " \
            "times.\n", \
            __FILE__, \
            __LINE__, \
            test_object.get_##mock_function##_counter_from_mock(), \
            number_of_expected_calls); \
        *result = false; \
        return; \
    } \
    else \
    { \
        *result = true; \
    }

#define GPU_ASSERT_INT_EQ(expected, actual) \
    if (expected != actual) \
    { \
        printf( \
            "Error detected on the GPU in file %s, line %d: Value %d returned, expected %d.\n", \
            __FILE__, \
            __LINE__, \
            actual, \
            expected); \
        *result = false; \
        return; \
    } \
    else \
    { \
        *result = true; \
    }

#define GPU_ASSERT_CLASS_EQ(expected, actual) \
    if (!(expected == actual)) \
    { \
        printf( \
            "Error detected on the GPU in file %s, line %d: Actual instance not equal to expected one.\n", \
            __FILE__, \
            __LINE__); \
        *result = false; \
        return; \
    } \
    else \
    { \
        *result = true; \
    }

#define GPU_ASSERT_NEAR(expected, actual, tolerance) \
    if (fabsf(expected - actual) > tolerance) \
    { \
        printf( \
            "Error detected on the GPU in file %s, line %d: Value %f returned, expected %f.\n", \
            __FILE__, \
            __LINE__, \
            actual, \
            expected); \
        *result = false; \
        return; \
    } \
    else \
    { \
        *result = true; \
    }

#define GPU_DEFINE_TEST_BODY(test_name, class_name) \
    class test_name##_class \
    { \
    public: \
        __device__ void executeTestBodyFor_##test_name(bool* result); \
    }; \
    __global__ void test_kernel_##test_name(bool* result) \
    { \
        *result = true; \
        test_name##_class device_object; \
        device_object.executeTestBodyFor_##test_name(result); \
    } \
    __device__ void test_name##_class::executeTestBodyFor_##test_name(bool* result)

#define GPU_DEFINE_TEST_BODY_UPLOAD(test_name, pointer_type, pointer) \
    class test_name##_class \
    { \
    public: \
        __device__ void executeTestBodyFor_##test_name(bool* result, pointer_type pointer); \
    }; \
    __global__ void test_kernel_##test_name(bool* result, pointer_type pointer) \
    { \
        *result = true; \
        test_name##_class device_object; \
        device_object.executeTestBodyFor_##test_name(result, pointer); \
    } \
    __device__ void test_name##_class::executeTestBodyFor_##test_name(bool* result, pointer_type pointer)

#define ADDITIONAL_LAUNCH_OPTIONS_FOR_CUDA_UNIT_TEST

#define GPU_TEST_EXECUTE_SET_OUTPUT(test_name, number_of_blocks, number_of_threads_per_block, profile_output) \
    bool* result_##test_name; \
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&result_##test_name, sizeof(bool))); \
    cudaEvent_t test_name##_start, test_name##_stop; \
    cudaEventCreate(&test_name##_start); \
    cudaEventCreate(&test_name##_stop); \
    cudaEventRecord(test_name##_start, 0); \
    test_kernel_##test_name<<< \
        number_of_blocks, \
        number_of_threads_per_block ADDITIONAL_LAUNCH_OPTIONS_FOR_CUDA_UNIT_TEST>>>(result_##test_name); \
    cudaEventRecord(test_name##_stop, 0); \
    cudaEventSynchronize(test_name##_stop); \
    float test_name##_kernel_time = 0.0f; \
    cudaEventElapsedTime(&test_name##_kernel_time, test_name##_start, test_name##_stop); \
    cudaEventDestroy(test_name##_start); \
    cudaEventDestroy(test_name##_stop); \
    cudaError error##test_name = cudaGetLastError(); \
    if (error##test_name != 0) \
        printf("CUDA error: %s\n", cudaGetErrorString(error##test_name)); \
    if (profile_output) \
        printf("Kernel execution in %s took %.3f microseconds.\n", #test_name, test_name##_kernel_time * 1000.f); \
    ASSERT_TRUE(*result_##test_name); \
    cudaFree(result_##test_name);

#define GPU_TEST_EXECUTE(test_name, number_of_blocks, number_of_threads_per_block) \
    GPU_TEST_EXECUTE_SET_OUTPUT(test_name, number_of_blocks, number_of_threads_per_block, false)

#define GPU_TEST_EXECUTE_WITH_PROFILING_OUTPUT(test_name, number_of_blocks, number_of_threads_per_block) \
    GPU_TEST_EXECUTE_SET_OUTPUT(test_name, number_of_blocks, number_of_threads_per_block, true)

#define GPU_TEST_EXECUTE_UPLOAD(test_name, number_of_blocks, number_of_threads_per_block, pointer) \
    bool* result_##test_name; \
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&result_##test_name, sizeof(bool))); \
    test_kernel_##test_name<<< \
        number_of_blocks, \
        number_of_threads_per_block ADDITIONAL_LAUNCH_OPTIONS_FOR_CUDA_UNIT_TEST>>>(result_##test_name, pointer); \
    cudaDeviceSynchronize(); \
    cudaError error##test_name = cudaGetLastError(); \
    if (error##test_name != 0) \
        printf("CUDA error: %s\n", cudaGetErrorString(error##test_name)); \
    ASSERT_TRUE(*result_##test_name); \
    cudaFree(result_##test_name);

#define GPU_FRIEND_TEST(test_name) friend class test_name##_class

#define GPU_GET_MOCK_COUNTER(object_name, mock_function) \
public: \
    __device__ int get_##mock_function##_counter_from_mock() \
    { \
        return object_name.count_##mock_function; \
    }
#endif
